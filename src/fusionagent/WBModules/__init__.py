import os
import sys
import yaml
from math import pi
from scipy.special import logsumexp
from functools import partial
import einops
# from timm.layers import trunc_normal_
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ParameterDict
from itertools import combinations
from abc import ABC, abstractmethod
from transformers import AutoModel, Qwen2_5_VLForConditionalGeneration
from huggingface_hub import hf_hub_download
import shutil

# from WBModules.CAL.test import extract_img_feature
# from WBModules.CAL.models.img_resnet import ResNet50
# from WBModules.CAL.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from WBModules.Biggait.modeling.models.BigGait import BigGait__Dinov2_Gaitbase as GaitModel
from WBModules.AGRL.torchreid.utils.reidtools import calc_splits
from data.transforms import get_transforms


MODEL_MAPPING_DICT = {'biggait': 'body_data',
                      'cal': 'body_data',
                      'cal-ccvid': 'body_data',
                      'cal-mevid': 'body_data',
                      'cal-ltcc': 'body_data',
                      'agrl': 'body_data',
                      'aim': 'body_data',
                      'kprpe': 'face_data',
                      'adaface': 'face_data',
                      'insightface': 'face_data',
                      'arcface': 'face_data'}

MODEL_DIM_KEYS = {'adaface': 512, 'kprpe': 512, 'arcface': 512, 'insightface': 512}

DEFAULT_BACKBONE_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'model_cfg_ccvid.yaml'), 'r'))

# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

            
# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

def build_face_backbone(backbone_cfg, mode='kprpe'):
    """
    ckpt_idr must be absolute path
    register the hook_fn for your face model to get the intermediate features from backbone blocks
    We only implement the hook_fn for adaface and kprpe right now
    """
    if 'adaface' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_adaface_vit_base_webface4m", 
                                      save_path=backbone_cfg['adaface_cache_path'], 
                                      HF_TOKEN=backbone_cfg['HF_TOKEN'])
        # register hook for intermediate features
        # first_hook_block = model.model.net.blocks[8] if hasattr(model.model, 'net') else model.model.model.net.blocks[8]
        # second_hook_block = model.model.net.blocks[16] if hasattr(model.model, 'net') else model.model.model.net.blocks[16]
        # first_hook_block.register_forward_hook(hook_fn)
        # second_hook_block.register_forward_hook(hook_fn)
    # elif 'insightface' in mode:
    #     from insightface.app import FaceAnalysis
    #     model = FaceAnalysis()
    #     print(f'face backbone model using {mode} model')
    #     return model
    elif 'kprpe' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_adaface_vit_base_kprpe_webface4m", 
                                        save_path=backbone_cfg['kprpe_cache_path'], 
                                        HF_TOKEN=backbone_cfg['HF_TOKEN'])
        aligner = load_model_by_repo_id('minchul/cvlface_DFA_mobilenet', backbone_cfg['kprpe_aligner_path'], HF_TOKEN=backbone_cfg['HF_TOKEN'])
        model.aligner = aligner
        # first_hook_block = model.model.net.blocks[8] if hasattr(model.model, 'net') else model.model.model.net.blocks[8]
        # second_hook_block = model.model.net.blocks[16] if hasattr(model.model, 'net') else model.model.model.net.blocks[16]
        # first_hook_block.register_forward_hook(hook_fn)
        # second_hook_block.register_forward_hook(hook_fn)
    elif 'arcface' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_arcface_ir101_webface4m", 
                                        save_path=backbone_cfg['arcface_cache_path'], 
                                        HF_TOKEN=backbone_cfg['HF_TOKEN'])
        
    else:
        raise NotImplementedError(f'not supported face model type: {mode}!')
    model.eval()
    model.mode = mode
    print(f'face backbone model using {mode} model')
    # print(f'load checkpoint from {ckpt_path}')
    return model

def build_vlm_backbone(backbone_cfg, mode='qwen'):
    """
    Build VLM backbone model.
    """
    if 'qwen2.5' in mode:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            backbone_cfg['qwen_model_name'],
            torch_dtype=torch.bfloat16,
            cache_dir=backbone_cfg['qwen_cache_path']
        )
    else:
        raise NotImplementedError(f'not supported vlm model type: {mode}!')
    model.eval()
    model.mode = mode
    print(f'vlm backbone model using {mode} model')
    return model

def build_gait_backbone(backbone_cfg, mode='biggait'):
    
    if 'biggait' in mode:
        ckpt_path = backbone_cfg['biggait_backbone_path']
        model_cfg_path = backbone_cfg['biggait_cfg_path']
        model_cfg = yaml.safe_load(open(model_cfg_path, 'r'))['model_cfg']
        model = GaitModel(model_cfg=model_cfg)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.mode = mode
    else:
        raise NotImplementedError(f'not supported gait model type: {mode}!')
    print(f'gait backbone model using {mode} model')
    print(f'load checkpoint from {ckpt_path}')
    return model

def build_body_backbone(backbone_cfg, mode='cal'):
    if 'cal' in mode:
        from WBModules.CAL.configs.default_img import _C as C_img
        from WBModules.CAL.configs.default_vid import _C as C_vid

        from WBModules.CAL.models.img_resnet import ResNet50
        from WBModules.CAL.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50

        factory = {
        'resnet50': ResNet50,
        'c2dres50': C2DResNet50,
        'i3dres50': I3DResNet50,
        'ap3dres50': AP3DResNet50,
        'nlres50': NLResNet50,
        'ap3dnlres50': AP3DNLResNet50,
        }
        if 'ccvid' in mode:
            config = C_img.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-ccvid_config_path'])
            ckpt_path = backbone_cfg['cal-ccvid_backbone_path']       
        elif 'mevid' in mode:
            config = C_vid.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-mevid_config_path'])
            ckpt_path = backbone_cfg['cal-mevid_backbone_path']
        elif 'ltcc' in mode:
            config = C_img.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-ltcc_config_path'])
            ckpt_path = backbone_cfg['cal-ltcc_backbone_path']
        else:
            raise NotImplementedError(f'not supported cal model type: {mode}!')
        model = factory[config.MODEL.NAME](config)
        model_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_dict)
        # if 'ccvid' in mode or 'ltcc' in mode:
        #     first_hook_block = model.base[5]
        #     second_hook_block = model.base[6]
        # elif 'mevid' in mode:
        #     first_hook_block = model.layer2
        #     second_hook_block = model.layer3
        # first_hook_block.register_forward_hook(hook_fn)
        # second_hook_block.register_forward_hook(hook_fn) 
    elif 'agrl' in mode:
        from WBModules.AGRL.torchreid import models
        model = models.init_model(name='vmgn', num_classes=104, loss={'xent', 'htri'},
                            last_stride=1, num_parts=3, num_scale=1,
                            num_split=4, pyramid_part=True, num_gb=2,
                            use_pose=False, learn_graph=True, consistent_loss=True,
                            bnneck=False, save_dir=os.path.join(os.path.dirname(__file__), 'AGRL'))
        ckpt_path = backbone_cfg['agrl_backbone_path']
        model_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        model.load_state_dict(model_dict)
    elif 'aim' in mode:
        from WBModules.AIM_CCReID.configs.default_img import _C as C_img
        from WBModules.AIM_CCReID.models.img_resnet import ResNet50
        config = C_img.clone()
        config.defrost()
        config.merge_from_file(backbone_cfg['aim_config_path'])
        model = ResNet50(config)
        ckpt_path = backbone_cfg['aim_backbone_path']
        model_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        raise NotImplementedError(f'not supported body model type: {mode}!')
    model.mode = mode
    model.eval()
    print(f'body backbone model using {mode} model')
    print(f'load checkpoint from {ckpt_path}')
    return model

class Model_Wrapper(nn.Module):
    """
    A wrapper class that builds different backbone models and handles image transformations.
    
    Args:
        mode (str): The model mode (e.g., 'kprpe', 'adaface', 'cal-ccvid', 'biggait', etc.)
        backbone_cfg (dict): Configuration dictionary for backbone models
        config (object, optional): Configuration object for transforms
        is_training (bool): Whether in training mode (affects transform selection)
    """
    
    def __init__(self, mode, backbone_cfg, config=None, is_training=False, dataset_type='vid'):
        super(Model_Wrapper, self).__init__()
        
        self.mode = mode
        self.backbone_cfg = backbone_cfg
        self.config = config
        self.is_training = is_training
        self.dataset_type = dataset_type
        
        # Build the appropriate backbone model
        self.model = self._build_backbone()
        
        # Build the appropriate transforms
        self._build_transform()
        
    def _build_backbone(self):
        """Build the backbone model based on the mode"""
        if self.mode in ['kprpe', 'adaface', 'arcface', 'insightface']:
            return build_face_backbone(self.backbone_cfg, self.mode)
        elif self.mode in ['qwen-vl']:
            return build_vlm_backbone(self.backbone_cfg, self.mode)
        elif self.mode in ['biggait']:
            return build_gait_backbone(self.backbone_cfg, self.mode)
        elif self.mode in ['cal-ccvid', 'cal-mevid', 'cal-ltcc', 'agrl', 'aim']:
            return build_body_backbone(self.backbone_cfg, self.mode)
        else:
            raise NotImplementedError(f'Unsupported model mode: {self.mode}')
    
    def _build_transform(self):
        """Build the appropriate transform based on the mode"""
        sp_transform_train, sp_transform_test, temp_transform_train, temp_transform_test = get_transforms(self.mode, self.config, dataset_type=self.dataset_type)
        self.spatial_transform_train = sp_transform_train
        self.spatial_transform_test = sp_transform_test
        self.temporal_transform_train = temp_transform_train
        self.temporal_transform_test = temp_transform_test
             
    def forward(self, pil_images):
        """
        Forward pass through the model
        
        Args:
            pil_images (list): List of PIL images
            
        Returns:
            torch.Tensor: Feature tensor of shape (N, D) where N is number of images
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        
        # Select appropriate transforms based on training mode
        if self.is_training:
            spatial_transform = self.spatial_transform_train
            temporal_transform = self.temporal_transform_train
        else:
            spatial_transform = self.spatial_transform_test
            temporal_transform = self.temporal_transform_test
        
        # Apply temporal transform if it exists (for video data)
        if temporal_transform is not None:
            # For PIL images input, temporal transform is not typically needed
            # but we keep this structure for consistency with dataset_loader
            mode_img_paths = pil_images
        else:
            mode_img_paths = pil_images
        
        # Apply spatial transforms
        clip = []
        if spatial_transform is not None:
            if hasattr(spatial_transform, 'randomize_parameters'):
                spatial_transform.randomize_parameters()
            clip = [spatial_transform(img) for img in mode_img_paths]
        else:
            # If no transform, convert PIL to tensor manually
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            clip = [to_tensor(img) for img in mode_img_paths]
        
        if len(clip) == 0:
            if self.is_training:
                # Create a zero tensor if no data is available
                clip = torch.zeros((3, len(mode_img_paths), 112, 112))
            else:
                clip = torch.zeros((0, 0, 0, 0))
            batch = clip
        else:
            # Stack images and handle dimension transformation
            batch = torch.stack(clip, 0)
        
        # Move to the same device as model
        device = next(self.model.parameters()).device
        batch = batch.to(device) # (t, c, h, w)
        
        if 'biggait' in self.mode:
            batch = batch.unsqueeze(0) # (1, t, c, h, w) for biggait, we need to add a dimension for the batch size
            if not self.is_training:
                with torch.no_grad():
                    features = self.model(([batch], [0], ['_'], ['_'], None))['inference_feat']['embeddings']  # [bs, c, n, p]
                    features = einops.rearrange(features, 'n c s p -> s (n c p)')
                    features = torch.max(features, dim=0)[0]  # [n, c, p]
                    features = einops.rearrange(features, '(n c p) -> n c p', n=1, c=512, p=16)
                    features = self.model.gait_net.FCs(features)  # [n, c//2, p]
                    features = features.flatten(1)  # (B, 4096)
            else:
                features = self.model(([batch], [0], ['_'], ['_'], None))['inference_feat']['embeddings']  # [bs, c, n, p]
                features = einops.rearrange(features, 'n c s p -> s (n c p)')
                features = torch.max(features, dim=0)[0]  # [n, c, p]
                features = einops.rearrange(features, '(n c p) -> n c p', n=1, c=512, p=16)
                features = self.model.gait_net.FCs(features)  # [n, c//2, p]
                features = features.flatten(1)  # (B, 4096)
        elif 'cal' in self.mode:
            if 'ccvid' in self.mode or 'ltcc' in self.mode:
                if not self.is_training:
                    with torch.no_grad():
                        features = self.model(batch)
                else:
                    features = self.model(batch)
            elif 'mevid' in self.mode:
                # video-based backbone (MEVID)
                batch = batch.unsqueeze(0).transpose(1, 2) # (1, t, c, h, w) -> (1, c, t, h, w)
                if not self.is_training:
                    with torch.no_grad():
                        features = self.model(batch)
                else:
                    features = self.model(batch)
        elif 'agrl' in self.mode:
            batch = batch.unsqueeze(0) # (1, t, c, h, w)
            bs, t, c, h, w = batch.shape
            # generate pose related graph for pretrained AGRL follow the paper MEVID
            adj_size = sum(calc_splits(4))
            adj_size = adj_size * t * 1
            adj = torch.ones((bs, adj_size, adj_size))
            if not self.is_training:
                with torch.no_grad():
                    features = self.model(batch, adj)
            else:
                features = self.model(batch, adj)
        elif 'aim' in self.mode:
            if not self.is_training:
                with torch.no_grad():
                    _, features = self.model(batch)
            else:
                _, features = self.model(batch)
        elif 'kprpe' in self.mode:
            # get face landmarks
            aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.model.aligner(batch)
            keypoints = orig_ldmks  # torch.randn(1, 5, 2)            
            if not self.is_training:
                with torch.no_grad():
                    features = self.model(batch, keypoints)
            else:
                features = self.model(batch, keypoints)
        elif 'qwen' in self.mode:
            if not self.is_training:
                with torch.no_grad():
                    # batch shape is (t, c, h, w)
                    vision_tower_output = self.model.vision_tower(batch)
                    # output is tuple, first element is last_hidden_state (t, seq_len, dim)
                    frame_features = vision_tower_output[0][:, 0, :] # CLS token for each frame -> (t, dim)
                    features = frame_features.mean(dim=0, keepdim=True) # (1, dim)
            else:
                # batch shape is (t, c, h, w)
                vision_tower_output = self.model.vision_tower(batch)
                # output is tuple, first element is last_hidden_state (t, seq_len, dim)
                frame_features = vision_tower_output[0][:, 0, :] # CLS token for each frame -> (t, dim)
                features = frame_features.mean(dim=0, keepdim=True) # (1, dim)
        
        else:
            # Forward through the backbone model
            if not self.is_training:
                with torch.no_grad():
                    features = self.model(batch)
            else:
                features = self.model(batch)
        
        return features
    
    def eval(self):
        """Set the model to evaluation mode"""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set the model to training mode"""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self


if __name__ == '__main__':
    
    HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'
    
    repo_id = 'minchul/cvlface_adaface_vit_base_webface4m'
    model = load_model_by_repo_id(repo_id, DEFAULT_BACKBONE_CFG['adaface_cache_path'], HF_TOKEN)
    
    # path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_kprpe_webface4m')
    # repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
    # model = load_model_by_repo_id(repo_id, path, HF_TOKEN)

    # input is a rgb image normalized.
    from torchvision.transforms import Compose, ToTensor, Normalize
    from PIL import Image
    img = Image.open('path/to/image.jpg')
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(img).unsqueeze(0)  # torch.randn(1, 3, 112, 112)
    
    # KPRPE also takes keypoints locations as input
    aligner = load_model_by_repo_id('minchul/cvlface_DFA_mobilenet', path, HF_TOKEN)
    aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(input)
    keypoints = orig_ldmks  # torch.randn(1, 5, 2)
    out = model(input, keypoints)
