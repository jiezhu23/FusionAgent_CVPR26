# Define the tool function
# We use get_json_schema(tool) to generate the description for the tool function
# All exection errors should be catched in the trainer.
# More details and format requirements: https://huggingface.co/docs/transformers/v4.52.3/chat_extras

from typing import List, Dict, Any, Literal
import torch
import torch.nn.functional as F

from WBModules import Model_Wrapper, MODEL_MAPPING_DICT, MODEL_DIM_KEYS
from utils.eval_metrics import evaluate_agent, score_fusion_method


def sim_fn(probe_feats, gallery_feats, model_name, norm_method='none'):
    """
    Compute similarity scores between probe embeddings and gallery embeddings for different backbone models.

    Args:
        probe_feats: A tensor of shape (P, D) containing probe embeddings.
        gallery_feats: A tensor of shape (G, D) containing gallery embeddings.
        model_name: The model identifier used to select the similarity formulation.
        norm_method: Optional normalization strategy applied to similarity scores. Defaults to 'none'.

    Returns:
        A tensor of shape (P, G) with similarity scores, where higher values indicate greater similarity.
    """
    # simiarity function for different models (e.g., cos_sim/euc_dist)
    if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name or 'cal' in model_name or 'agrl' in model_name or 'aim' in model_name:
        scores = F.cosine_similarity(probe_feats.unsqueeze(1), gallery_feats.unsqueeze(0), dim=2)
    elif 'biggait' in model_name:
        bs = probe_feats.shape[0]
        num_g = gallery_feats.shape[0]
        # euc distance
        gait_score = torch.norm(gallery_feats.reshape(num_g, -1, 16).unsqueeze(0) - probe_feats.reshape(bs, -1, 16).unsqueeze(1), p=2, dim=2) #(P, G, 16)
        gait_score = gait_score.mean(dim=-1) #(P, G)
        scores = 1 / (1 + gait_score)
    else:
        raise ValueError(f'unknown model name: {model_name}')
    if norm_method == 'none':
        scores = scores
    return scores

def get_embedding(sequence: dict, model: Model_Wrapper):
    """
    Compute a sequence-level embedding by averaging per-frame embeddings from the provided model.

    Args:
        sequence: A dictionary with 'body_pil_image' or 'face_pil_image' available.
        model: A Model_Wrapper instance used to extract embeddings.

    Returns:
        A dictionary with key 'embedding', containing a tensor of shape (1, D) representing the sequence embedding.
        We use average pooling to get the sequence-level embedding, except some video-based models (e.g., 'biggait', 'agrl', 'cal-mevid').
    """
    model_type = MODEL_MAPPING_DICT[model.mode]
    if model_type == 'face_data':
        # handle empty face inside the sequence
        inputs = sequence['face_pil_image']
        if len(inputs) == 0 or all(x is None for x in inputs):
            return {'embedding': torch.zeros(1, MODEL_DIM_KEYS[model.mode])}
        outs = model(inputs) # (N', D)
    else:
        inputs = sequence['body_pil_image']
        if len(inputs) == 0 or all(x is None for x in inputs):
            return {'embedding': torch.zeros(1, MODEL_DIM_KEYS[model.mode])}
        outs = model(inputs) # (N, D)
    # average over the frames
    outs = outs.mean(dim=0, keepdim=True) # (1, D)
    
    return {'embedding': outs}

def get_similarity_matrix(sequence: dict, model: Model_Wrapper, center_feats: torch.Tensor, model_name: str):
    """
    Compute the similarity matrix between a sequence embedding and provided center features.

    Args:
        sequence: A dictionary with 'body_pil_image' or 'face_pil_image' available.
        model: A Model_Wrapper instance used to extract embeddings.
        center_feats: A tensor of shape (G, D) containing gallery/center embeddings.
        model_name: The model identifier used to select the similarity formulation.

    Returns:
        A dictionary with keys:
            - 'similarity_matrix': A tensor of shape (1, G) with similarity scores.
            - 'embedding': A tensor of shape (1, D) representing the sequence embedding.
    """
    sequence_embeddings = get_embedding(sequence, model)['embedding'] # (1, D)
    # Ensure center_feats is on the same device as sequence_embeddings
    center_feats = center_feats.to(sequence_embeddings.device)
    similarity_matrix = sim_fn(sequence_embeddings, center_feats, model_name) # (1, G)
    
    result = {
        'similarity_matrix': similarity_matrix,
        'embedding': sequence_embeddings,
    }
    return result

def get_result(sequence: dict, model_list: List[Model_Wrapper], center_feats: dict, model_name: str):
    """
    Evaluate the sequence against center features and return ranking metrics and embeddings.
    Only support model name in ['adaface', 'arcface', 'cal', 'agrl', 'aim', 'kprpe', 'biggait'].
    center_feats and model_list would be automatically provided.

    Args:
        sequence: Leave it empty, it would be automatically provided.
        model_list: Leave it empty, it would be automatically provided.
        center_feats: Leave it empty, it would be automatically provided.
        model_name: The key to select the backbone from model_list and its corresponding center features.

    Returns:
        A dictionary including:
            - 'Rank1': Rank-1 accuracy range in [0, 1] (float)
            - 'mAP': mean Average Precision range in [0, 1] (float)
            - 'TAR@1.00%FAR': True Accept Rate at 1% False Accept Rate range in [0, 1] (float)
            - 'overall': Aggregated score across metrics (float)
            - 'embedding': The sequence embedding tensor of shape (1, D)
            - 'similarity_matrix': The similarity tensor of shape (1, G)
    """
    model = model_list[model_name]
    sim_result = get_similarity_matrix(sequence, model, center_feats['center_feats'][model_name], model_name)
    q_pids = torch.tensor([int(sequence['subject_id'].split('_')[1])])
    g_pids = center_feats['center_pids'].cpu()
    eval_result = evaluate_agent(sim_result['similarity_matrix'].cpu(), q_pids, g_pids, far=[0.01])
    
    result = {
        'Rank1': eval_result['Rank1'],
        'mAP': eval_result['mAP'],
        r'TAR@1.00%FAR': eval_result[r'TAR@1.00%FAR'],
        'identity': str(center_feats['center_pids'][sim_result['similarity_matrix'].argmax()].item()),
        'similarity_score': sim_result['similarity_matrix'].max().item(),
        'embedding': sim_result['embedding'].cpu(),
        'similarity_matrix': sim_result['similarity_matrix'].cpu(),
    }
    return result
    

#-------------------- Agent Tools --------------------
class RuntimeCtx:
    """
    Runtime context holder for tool execution with flexible key-value storage.

    Hidden parameters that should NOT be exposed to the agent (thus not part
    of the tool function schema) are stored here and read by tool wrappers.
    
    Supports dynamic key-value storage for extensibility.
    """
    # Core system parameters (backward compatibility)
    sequence: dict | None = None
    model_list: Dict[str, Model_Wrapper] | None = None
    center_feats: dict | None = None
    dataset_name: str | None = None
    train_feats: dict | None = None
    test_feats: dict | None = None
    # Dynamic key-value storage for extensible system-level data
    _storage: Dict[str, Any] = {}

    @classmethod
    def set(
        cls,
        *,
        sequence: dict | None = None,
        model_list: Dict[str, Model_Wrapper] | None = None,
        center_feats: dict | None = None,
        dataset_name: str | None = None,
        train_feats: dict | None = None,
        test_feats: dict | None = None,
    ) -> None:
        """Set core system parameters (backward compatibility)."""
        if sequence is not None:
            cls.sequence = sequence
        if model_list is not None:
            cls.model_list = model_list
        if center_feats is not None:
            cls.center_feats = center_feats
        if dataset_name is not None:
            cls.dataset_name = dataset_name
        if train_feats is not None:
            cls.train_feats = train_feats
        if test_feats is not None:
            cls.test_feats = test_feats

    @classmethod
    def put(cls, key: str, value: Any) -> None:
        """
        Store a key-value pair in the runtime context.
        
        Args:
            key: The key to store the value under
            value: The value to store (can be any type)
        """
        cls._storage[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by key from the runtime context.
        
        Args:
            key: The key to retrieve
            default: Default value to return if key doesn't exist
            
        Returns:
            The stored value or default if key doesn't exist
        """
        return cls._storage.get(key, default)

    @classmethod
    def delete(cls, key: str) -> bool:
        """
        Delete a key-value pair from the runtime context.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key in cls._storage:
            del cls._storage[key]
            return True
        return False

    @classmethod
    def has(cls, key: str) -> bool:
        """
        Check if a key exists in the runtime context.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in cls._storage

    @classmethod
    def keys(cls) -> List[str]:
        """
        Get all keys in the runtime context storage.
        
        Returns:
            List of all keys in storage
        """
        return list(cls._storage.keys())

    @classmethod
    def clear_storage(cls) -> None:
        """Clear all key-value pairs from storage."""
        cls._storage.clear()

    @classmethod
    def get_all_storage(cls) -> Dict[str, Any]:
        """
        Get a copy of all stored key-value pairs.
        
        Returns:
            Dictionary containing all stored key-value pairs
        """
        return cls._storage.copy()

    @classmethod
    def clear_all(cls) -> None:
        """Clear all key-value pairs from storage."""
        cls._storage.clear()
        cls.sequence = None
        cls.model_list = None
        cls.center_feats = None
        cls.dataset_name = None

    @classmethod
    def fuse_similarity_matrix(cls, fuse_method: str = 'mean') -> torch.Tensor:
        """Find all similarity matrices in the storage and fuse them."""
        similarity_matrices = []
        for key in cls.keys():
            if 'similarity_matrix' in key:
                similarity_matrices.append(cls.get(key))
        
        if not similarity_matrices:
            return None
        
        # Ensure all tensors are on the same device
        device = similarity_matrices[0].device
        similarity_matrices = [matrix.to(device) for matrix in similarity_matrices]
        
        fused_matrix = torch.cat(similarity_matrices, dim=0)
        
        if fuse_method == 'mean':
            fused_result = fused_matrix.mean(dim=0, keepdim=True)
        elif fuse_method == 'max':
            fused_result = fused_matrix.max(dim=0, keepdim=True)
        else:
            raise ValueError(f"Invalid fuse_method: {fuse_method}")
        
        cls.put('fused_matrix', fused_result)
        return fused_result

def get_result_tool(
    model_name: str
) -> dict:
    """
    Evaluate the sequence against center features and return ranking metrics.

    Args:
        model_name: The backbone to use.

    Returns:
        A dictionary including:
            - 'pred_identity_id': Predicted identity ID (str)
            - 'model_name': The model name used (str)
    """
    # Runtime validation to keep dynamic allowed list at run time
    allowed_models = list(RuntimeCtx.model_list.keys()) if RuntimeCtx.model_list else None
    if allowed_models is not None and model_name not in allowed_models:
        raise ValueError(f"Invalid model_name: {model_name}. Allowed: {sorted(allowed_models)}")
    
    model = RuntimeCtx.model_list[model_name]
    sim_mat = get_similarity_matrix(RuntimeCtx.sequence, model, RuntimeCtx.center_feats['center_feats'][model_name], model_name)
    # Store system-level results using the flexible key-value API
    RuntimeCtx.put(f'{model_name}_embedding', sim_mat['embedding'])
    RuntimeCtx.put(f'{model_name}_similarity_matrix', sim_mat['similarity_matrix'])

    # Post-process to get the fused result from current model and other models' results
    # RuntimeCtx.fuse_similarity_matrix(fuse_method='mean')
    # fused_matrix = RuntimeCtx.get('fused_matrix').cpu()
    # q_pids = torch.tensor([int(RuntimeCtx.sequence['subject_id'].split('_')[1])])
    # g_pids = RuntimeCtx.center_feats['center_pids'].cpu()
    # fused_result = evaluate_agent(fused_matrix, q_pids, g_pids, far=[0.01])
    # performance_metrics = (fused_result['Rank1'] + fused_result['mAP'] + fused_result[r'TAR@1.00%FAR']) / 3
    
    # check if the similarity matrix is all zeros
    if torch.all(sim_mat['embedding'] == 0):
        agent_result = {'error': 'No detected features in the sequence.', 'model_name': model_name}
    else:
        pred_identity_id = str(RuntimeCtx.center_feats['center_pids'][sim_mat['similarity_matrix'].argmax()].item())
        # Return only metric information to the agent
        agent_result = {
            'pred_identity_id': f'{RuntimeCtx.dataset_name}_{pred_identity_id}',
            'model_name': model_name,
            # 'confidence_score': round(performance_metrics, 3),
        }
    
    return agent_result