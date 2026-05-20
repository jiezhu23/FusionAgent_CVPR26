# Define the reward function
from typing import List, Union
import numpy as np
import torch

from fusionagent.utils import ConversationParser
from fusionagent.utils.eval_metrics import test_score, normalize_score, act_score_fusion

API_URL = "http://127.0.0.1:8001/v1"

def soft_overlong_punishment(
    completion_length: Union[int, float, List[Union[int, float]]], 
    L_max: Union[int, float] = 250, 
    L_cache: Union[int, float] = 50, 
    **kwargs
) -> Union[float, List[float]]:
    """Apply soft punishment for overly long completions.

    Args:
        completion_length: List of completion lengths or single length
        L_max: Maximum allowed length (default: 250)
        L_cache: Cache length for soft punishment (default: 50)
        
    Returns:
        List of rewards or single reward
    """
    # Handle single value or list
    if isinstance(completion_length, (int, float)):
        completion_lengths = [completion_length]
        return_single = True
    else:
        completion_lengths = completion_length
        return_single = False
    
    rewards = []
    for cl in completion_lengths:
        if cl <= L_max - L_cache:
            rewards.append(0)
        elif L_max - L_cache < cl <= L_max:
            rewards.append((L_max - L_cache - cl) / L_cache)
        else:
            rewards.append(-1)
    
    return rewards[0] if return_single else rewards


def conversation_format_reward(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that checks if each assistant turn has complete format.
    
    The format check depends on the `prompt_type` argument in kwargs.
    prompt_type: [fast, cot]. fast means fast mode, cot means cooperative thinking mode.
    If prompt_type == 'cot':
        Complete format means the turn content matches either:
        1. <think>...</think> followed by <tool_call>...</tool_call>
        2. <think>...</think> followed by <answer>...</answer>
    Else:
        Complete format means the turn content matches either:
        1. <tool_call>...</tool_call>
        2. <answer>...</answer>
    
    For each turn, if the response is matched the format, the reward is 1.0. Otherwise, the reward is 0.0.
    The reward of the whole conversation is the average reward of all turns.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        Average format completeness across all assistant turns (0.0 to 1.0)
    """
    prompt_type = kwargs.get("prompt_type", "cot")
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        total_turns = parser.get_total_turns()
        
        if total_turns == 0:
            rewards.append(0.0)
            continue
        
        correct_formats = 0
        for i in range(total_turns):
            if prompt_type == 'cot':
                is_correct = parser.has_complete_format(i)
            else:
                is_correct = parser.has_complete_format_fast(i)

            if is_correct:
                correct_formats += 1
        
        rewards.append(correct_formats / total_turns)
    
    return rewards[0] if return_single else rewards


def conversation_format_reward_v2(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that checks if each assistant turn has a complete and strict format.
    
    The format check depends on the `prompt_type` argument in kwargs.
    prompt_type: [fast, cot]. fast means fast mode, cot means cooperative thinking mode.

    If prompt_type == 'cot':
        Strict complete format means the turn content matches either:
        1. <think>...</think> followed by <tool_call>...</tool_call>
        2. <think>...</think> followed by <answer>...</answer>
        and no other tags are present at the top level.
    Else:
        Strict complete format means the turn content matches either:
        1. <tool_call>...</tool_call>
        2. <answer>...</answer>
        and no other tags are present at the top level.
    
    For each turn, if the response is matched the format, the reward is 1.0. Otherwise, the reward is 0.0.
    The reward of the whole conversation is the average reward of all turns.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        Average format completeness across all assistant turns (0.0 to 1.0)
    """
    prompt_type = kwargs.get("prompt_type", "cot")
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        total_turns = parser.get_total_turns()
        
        if total_turns == 0:
            rewards.append(0.0)
            continue
        
        correct_formats = 0
        for i in range(total_turns):
            if prompt_type == 'cot':
                is_correct = parser.has_complete_format_v2(i)
            else:
                is_correct = parser.has_complete_format_fast_v2(i)
            
            if is_correct:
                correct_formats += 1
        
        rewards.append(correct_formats / total_turns)
    
    return rewards[0] if return_single else rewards


def tool_success_rate_reward(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that calculates tool call success rate.
    
    Success is determined by tool_result not containing an 'error' key.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        Success rate of tool calls (0.0 to 1.0)
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        tool_success_rate = parser.get_tool_success_rate()
        rewards.append(tool_success_rate)
    
    return rewards[0] if return_single else rewards


def multi_turn_reward(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    min_turn: int = 3, 
    max_turn: int = 4, 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that checks if conversation length is within acceptable range.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        min_turn: Minimum acceptable number of assistant turns
        max_turn: Maximum acceptable number of assistant turns
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        1.0 if turn count is within range, 0.0 otherwise
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        total_turns = parser.get_total_turns()
        
        if min_turn <= total_turns <= max_turn:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards[0] if return_single else rewards


def answer_accuracy_reward(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    labels: List[str],
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that checks if the predicted answer is the gt label.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        1.0 if the predicted answer is in the final answer, 0.0 otherwise
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser, label in zip(parsers, labels):
        # 1. Check if the last assistant turn has an <answer> tag
        if not parser.assistant_turns or not parser.parsed_turns:
            rewards.append(0.0)
            continue

        last_turn_parser = parser.parsed_turns[-1]
        if not last_turn_parser:
            rewards.append(0.0)
            continue
            
        answer_content = last_turn_parser.get_first_content('answer')
        if not answer_content:
            rewards.append(0.0)
            continue

        # 2. Check if the answer is the gt label
        rewards.append(1.0 if label in answer_content else 0.0)
        
    return rewards[0] if return_single else rewards


def metric_reward(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    test_feats: dict,
    train_feats: dict,
    top_k: int,
    main_frac: float,
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that gets the model combination from the conversation and evaluate the performance on test set.

    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        # 1. Check if the last assistant turn has an <answer> tag
        if not parser.assistant_turns or not parser.parsed_turns:
            rewards.append(0.0)
            continue
        selected_model_list = []
        for turn in parser.parsed_turns:
            if 'tool_call' in turn.get_ordered_tags():
                try:
                    tool_input = turn.parse_tool_calls()
                    model_name = tool_input[0]['tool_kwargs']['model_name']
                    selected_model_list.append(model_name)
                except Exception:
                    continue
        if len(selected_model_list) == 0:
            rewards.append(0.0)
            continue
        try:
            # TODO: other fusion algorithms?
            # -----------------------confidence top-k score fusion-----------------------
            input_feats = train_feats
            norm_method = 'zscore'
            anchor_model = selected_model_list[0] # the first model is the anchor model
            top_k = top_k
            main_frac = main_frac
            model_list = list(input_feats['score_mats'].keys())
            selected_model_mask = np.array([1 if model_name in selected_model_list else 0 for model_name in model_list])
            anchor_model_index = model_list.index(anchor_model)
            score_mats = {model_name: input_feats['score_mats'][model_name].cpu().numpy() for model_name in model_list}
            merge_score_mats = {model_name: input_feats['merge_score_mats'][model_name].cpu().numpy() for model_name in model_list}
                
            # create a random model_mask based on the selected model_list
            model_mask = np.zeros((score_mats[model_list[0]].shape[0], len(model_list)), dtype=bool)
            model_mask[:int(score_mats[model_list[0]].shape[0]*main_frac)] = selected_model_mask
            # set the rest of the model_mask to be random
            model_mask[int(score_mats[model_list[0]].shape[0]*main_frac):] = np.random.randint(0, 2, (score_mats[model_list[0]].shape[0]-int(score_mats[model_list[0]].shape[0]*main_frac), len(model_list)), dtype=bool)
            # make sure the anchor model is used
            model_mask[:, anchor_model_index] = True
            # randomly shuffle the model_mask
            model_mask = np.random.permutation(model_mask)
            
            anchor_model_list = np.full(model_mask.shape[0], anchor_model, dtype=object)
            fused_score_mat, fused_merge_score_mat = act_score_fusion(score_mats, merge_score_mats, model_mask, anchor_model_list, top_k=top_k, norm_method=norm_method)
            
            # ------------------------------mean score fusion-----------------------------
            # score_mats = [test_feats['score_mats'][model_name] for model_name in selected_model_list]
            # merge_score_mats = [test_feats['merge_score_mats'][model_name] for model_name in selected_model_list]
            # fused_score_mat = torch.stack(score_mats, dim=1).mean(dim=1).cpu().numpy() # (B, G)
            # fused_merge_score_mat = torch.stack(merge_score_mats, dim=1).mean(dim=1).cpu().numpy() # (B, G)
            
            # We do not fix seed for reward function to increase the diversity of the reward
            fused_result = test_score(fused_score_mat, fused_merge_score_mat, 
                                    input_feats['q_pids'], input_feats['q_camids'], input_feats['q_clothes_ids'], 
                                    input_feats['g_pids'], input_feats['g_camids'], input_feats['g_clothes_ids'], input_feats['unique_g_pids'], 
                                    dataset=input_feats['dataset_name'], log=False)
            # reward = (fused_result[r'TAR@1.00%FAR'] + fused_result['GR_top1'] + fused_result['GR_mAP'] - fused_result[r'FNIR@1.00%FPIR'])
            reward = (fused_result['GR_top1']) # for the rebuttal

        except Exception as e:
            print(e)
            rewards.append(0.0)
            continue
        rewards.append(reward)
    return rewards[0] if return_single else rewards


def metric_reward_v0(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that gets the final metric score from the last tool result.

    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        # 1. Check if the last assistant turn has an <answer> tag
        if not parser.assistant_turns or not parser.parsed_turns:
            rewards.append(0.0)
            continue
        # get the last tool result
        confidence_score = 0.0
        for turn in reversed(parser.conversation[2:]):
            if turn.get('role') == 'system' and '<tool_result>' in turn.get('content', ''):
                content = turn.get('content', '')
                start = content.find('<tool_result>') + len('<tool_result>')
                end = content.find('</tool_result>')
                result_content = content[start:end].strip()
                # Try to parse as dictionary
                result_data = eval(result_content)
                if isinstance(result_data, dict) and 'confidence_score' in result_data:
                    confidence_score = result_data['confidence_score']
                    break
            else:
                continue
        rewards.append(confidence_score)
    
    return rewards[0] if return_single else rewards


def answer_accuracy_reward_v0(
    conversations: Union[ConversationParser, List[ConversationParser]], 
    **kwargs
) -> Union[float, List[float]]:
    """Reward function that checks if the predicted answer is in the final answer.
    
    Args:
        conversations: Single ConversationParser or list of ConversationParser objects
        
    Returns:
        Single reward (float) or list of rewards (List[float])
        1.0 if the predicted answer is in the final answer, 0.0 otherwise
    """
    # Handle both single parser and multiple parsers
    if isinstance(conversations, list):
        parsers = conversations
        return_single = False
    else:
        parsers = [conversations]
        return_single = True
    
    rewards = []
    for parser in parsers:
        # 1. Check if the last assistant turn has an <answer> tag
        if not parser.assistant_turns or not parser.parsed_turns:
            rewards.append(0.0)
            continue

        last_turn_parser = parser.parsed_turns[-1]
        if not last_turn_parser:
            rewards.append(0.0)
            continue
            
        answer_content = last_turn_parser.get_first_content('answer')
        if not answer_content:
            rewards.append(0.0)
            continue

        # 2. Find all 'pred_identity_id' from all tool_results
        pred_ids = []
        # skip the first two turns (system and user)
        for turn in parser.conversation[2:]:
            if turn.get('role') == 'system' and '<tool_result>' in turn.get('content', ''):
                content = turn.get('content', '')
                start = content.find('<tool_result>') + len('<tool_result>')
                end = content.find('</tool_result>')
                result_content = content[start:end].strip()
                
                try:
                    result_data = eval(result_content)
                    if isinstance(result_data, dict) and 'pred_identity_id' in result_data:
                        pred_id = result_data.get('pred_identity_id')
                        if pred_id is not None:
                            pred_ids.append(str(pred_id))
                except Exception:
                    continue
        
        # 3. Check if any pred_id is in the answer
        found = any(pred_id in answer_content for pred_id in pred_ids)
        rewards.append(1.0 if found else 0.0)
        
    return rewards[0] if return_single else rewards