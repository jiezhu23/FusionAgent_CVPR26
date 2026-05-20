import h5py
import numpy as np
import torch
from WBModules import MODEL_DIM_KEYS

def load_center_feat_subject(h5_file, labels, model_name):
    # load subject-level center feat
    f = h5py.File(h5_file, 'r')
    center_feat = []
    for id in labels:
        if id in f:
            center_feat.append(f[id][:])
        else:
            # subject 102 in MEVID does not have face imgs
            center_feat.append(np.zeros((1, MODEL_DIM_KEYS[model_name]), dtype=np.float32))
    center_feat = np.array(center_feat).squeeze()
    return center_feat

def load_center_feat_cc(h5_file, labels, model_name):
    # load clothing-level center feat 
    f = h5py.File(h5_file, 'r')
    center_feat, center_pids,  = [], []
    center_keys = []
    for id in labels:
        if id in f:
            camids_clothesids = sorted(list(f[id].keys()))
            for cid in camids_clothesids:
                center_feat.append(f[id][cid][:])
                center_pids.append(int(id))
                center_keys.append(f'{id}_{cid}')

        # else:
        #     # subject 102 in MEVID does not have face imgs
        #     center_feat.append(np.zeros((1, MODEL_DIM_KEYS[model_name]), dtype=np.float32))
        #     center_pids.append(int(id))
        #     center_camids.append(int(cid.split('_')[0]))
        #     center_clothes_ids.append(int(cid.split('_')[1]))
            
    center_feat = np.array(center_feat).squeeze()
    center_pids = np.array(center_pids)
    # center_camids = np.array(center_camids)
    # center_clothes_ids = np.array(center_clothes_ids)
    center_keys = np.array(center_keys)
    return center_feat, center_pids, center_keys

def load_center_feat(model_list, dataset, device='cpu', few_shot=None):
    # load training set center features
    center_feats = {}
    center_pids = {}
    center_keys = {}
    for model_name in model_list:
        if few_shot is None:
            h5_file = f'./src/fusionagent/mod_center_feat/{model_name}_center_{dataset.dataset_name}_CC.h5'
        else:
            h5_file = f'./src/fusionagent/mod_center_feat/{model_name}_center_{dataset.dataset_name}_fewshot{few_shot}.h5'
        _center_feats, _center_pids, _center_keys= load_center_feat_cc(h5_file, 
                                                    [str(i) for i in range(dataset.num_train_pids)], model_name)
        center_feats[model_name] = torch.tensor(_center_feats).to(device)
        center_pids[model_name] = _center_pids
        center_keys[model_name] = _center_keys
        
    # make sure center_labels are the same
    if not all(len(center_pids[model_name]) == len(center_pids[model_list[0]]) for model_name in model_list):
        ref_model_idx = np.array([len(center_pids[model_name]) for model_name in model_list]).argmax()
        ref_model_name = model_list[ref_model_idx]
        for model_name in model_list:
            if center_feats[model_name].shape[0] != center_feats[ref_model_name].shape[0]:
                _center_feats = torch.zeros((center_feats[ref_model_name].shape[0], center_feats[model_name].shape[1]), device=device)
                _center_pids = np.empty_like(center_pids[ref_model_name])
                _center_keys = np.empty_like(center_keys[ref_model_name])
                for i, (pids, keys) in enumerate(zip(center_pids[ref_model_name], center_keys[ref_model_name])):
                    if keys in center_keys[model_name]:
                        _center_feats[i] = center_feats[model_name][center_keys[model_name] == keys]
                    _center_pids[i] = pids
                    _center_keys[i] = keys
                center_feats[model_name] = _center_feats
                center_pids[model_name] = _center_pids
                center_keys[model_name] = _center_keys
    center_pids = torch.tensor(center_pids[model_list[0]]).to(device)
    center_camids = torch.tensor([int(k.split('_')[1]) for k in center_keys[model_list[0]]]).to(device)
    center_clothes_ids = torch.tensor([int(k.split('_')[2]) for k in center_keys[model_list[0]]]).to(device)
    center_feats = {
        'center_feats': center_feats,
        'center_pids': center_pids,
        'center_camids': center_camids,
        'center_clothes_ids': center_clothes_ids,
    }
    return center_feats


