import h5py
import yaml
import time
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import os.path as osp
import torch
import torch.nn.functional as F
from torch import distributed as dist
from transformers import set_seed

from data import build_dataloader, VID_DATASET
from WBModules import Model_Wrapper, MODEL_DIM_KEYS
from WBModules.qme import QME
from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from tool_func import sim_fn
from utils.eval_metrics import *
from utils.load_center_utils import load_center_feat

FILE_ROOT = osp.dirname(osp.abspath(__file__))


DATASET_MAPPING = {
    'ccvid': ['adaface', 'biggait', 'cal-ccvid', 'kprpe'],
    'mevid': ['adaface', 'agrl', 'cal-mevid', 'kprpe'],
    'ltcc': ['adaface', 'aim', 'cal-ltcc', 'kprpe']
}

def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(model, dataloader):
    logger = logging.getLogger('reid.test')
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
        if (batch_idx + 1) % 100==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        imgs = {k: v.cuda() for k, v in imgs.items()}
        batch_features, _ = model.get_feats(imgs)  # (B, d)
        batch_features = batch_features[model.model_list[0]]
        if batch_features is None:
            # NaN feature if no face is detected
            batch_features = torch.full((1, MODEL_DIM_KEYS[model.model_list[0]]), float('nan'))
        # flip_imgs = torch.flip(imgs, [3])
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        # batch_features = model(imgs)
        # batch_features_flip = model(flip_imgs)
        # batch_features += batch_features_flip
        # batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length, num_data=None):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    logger = logging.getLogger('reid.test')
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    limit_batches = num_data is not None and vid2clip_index is None
    if limit_batches:
        processed_count = 0

    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
        if limit_batches and processed_count >= num_data:
            break
        # if (batch_idx + 1) % 100==0:
        #     logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = {k: v.cuda() for k, v in vids.items()}
        batch_features, _ = model.get_feats(vids)  # (B, d)
        batch_features = batch_features[model.model_list[0]]
        if batch_features is None:
            # NaN feature if no face is detected
            batch_features = torch.full((1, MODEL_DIM_KEYS[model.model_list[0]]), float('nan'))
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
        if limit_batches:
            processed_count += len(batch_pids)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    if limit_batches:
        data_length = processed_count
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    if vid2clip_index is None:
        # assert we are extracting features for train_dense
        return clip_features, clip_pids, clip_camids, clip_clothes_ids
    else:
        # Use the averaged feature of all clips split from a video as the representation of this original full-length video
        features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
        clip_features = clip_features.cuda()
        pids = torch.zeros(len(vid2clip_index))
        camids = torch.zeros(len(vid2clip_index))
        clothes_ids = torch.zeros(len(vid2clip_index))
        for i, idx in enumerate(vid2clip_index):
            valid_mask = ~torch.isnan(clip_features[idx[0] : idx[1], :])
            if valid_mask.any():
                valid_feats = clip_features[idx[0] : idx[1], :][valid_mask].view(-1, clip_features.size(1))
                features[i] = valid_feats.mean(0)
            else:
                # NaN feature if no face is detected for all clips
                features[i] = torch.full((clip_features.size(1),), float('nan'))
                
            # features[i] = clip_features[idx[0] : idx[1], :].mean(0)
            # features[i] = F.normalize(features[i], p=2, dim=0)
            pids[i] = clip_pids[idx[0]]
            camids[i] = clip_camids[idx[0]]
            clothes_ids[i] = clip_clothes_ids[idx[0]]
        features = features.cpu()

        return features, pids, camids, clothes_ids


def extract_test_feats(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query['dataset']))
        torch.cuda.empty_cache()
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery['dataset']))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query['dataset']))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery['dataset']))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    # save extracted features
    fname = osp.join(FILE_ROOT, 'test_feats', f"{'_'.join(model.model_list)}_{config.DATA.DATASET}_test.h5")
    f = h5py.File(fname, 'w')
    f['qf'] = qf.cpu()
    f['q_pids'] = q_pids.cpu()
    f['q_camids'] = q_camids.cpu()
    f['q_clothes_ids'] = q_clothes_ids.cpu()
    f['gf'] = gf.cpu()
    f['g_pids'] = g_pids.cpu()
    f['g_camids'] = g_camids.cpu()
    f['g_clothes_ids'] = g_clothes_ids.cpu()
    f.close()
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return fname


def extract_train_feats(config, model, train_dataloader, dataset, saved=False, num_data=None):
    logger = logging.getLogger('reid.extract_train_feats')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        # use train_dense to extract features
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, train_dataloader, 
                                                                  None,
                                                                  len(dataset.train_dense['dataset']),
                                                                  num_data=num_data)
        torch.cuda.empty_cache()
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, train_dataloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query['dataset']))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    # save extracted featuresd
    if saved:
        fname = osp.join(FILE_ROOT, 'train_feats', f"{'_'.join(model.model_list)}_{config.DATA.DATASET}_train.h5")
        os.makedirs(osp.dirname(fname), exist_ok=True)
        f = h5py.File(fname, 'w')
        f['qf'] = qf.cpu()
        f['q_pids'] = q_pids.cpu()
        f['q_camids'] = q_camids.cpu()
        f['q_clothes_ids'] = q_clothes_ids.cpu()
        f.close()
    
    logger.info("Extracted features for training set, obtained {} matrix".format(qf.shape))    
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return qf, q_pids, q_camids, q_clothes_ids


def get_scoremat_from_feats(data_dict, mode='biggait'):
    # read extracted features
    qf, q_pids, q_camids, q_clothes_ids = data_dict['qf'], data_dict['q_pids'], data_dict['q_camids'], data_dict['q_clothes_ids']
    gf, g_pids, g_camids, g_clothes_ids = data_dict['gf'], data_dict['g_pids'], data_dict['g_camids'], data_dict['g_clothes_ids']
    
    # Get tracklet-level similarity score matrix
    if not isinstance(qf, torch.Tensor):
        qf = torch.tensor(qf)
    if not isinstance(gf, torch.Tensor):
        gf = torch.tensor(gf)
    m, n = qf.size(0), gf.size(0)
    score_mat = torch.zeros((m,n))
    for i in range(m):
        score_mat[i] = sim_fn(qf[None, i], gf, mode)
    
    # NaN feature will be replaced with 0
    score_mat = np.nan_to_num(score_mat.numpy())
    
    # Get subject-level similarity score matrix
    # find the same g_pid in g_pids then merge gallery features
    unique_g_pids = np.unique(g_pids)
    unique_gf = np.zeros((len(unique_g_pids), gf.shape[1]))
    for i, pid in enumerate(unique_g_pids):
        idx = np.where(g_pids == pid)[0]
        # merge gallery feature using average pooling
        unique_gf[i] = np.nanmean(gf[idx], axis=0, keepdims=True)

    unique_gf = torch.from_numpy(unique_gf)
    merge_score_mat = torch.zeros((m, len(unique_g_pids)))
    for i in range(m):
        merge_score_mat[i] = sim_fn(qf[None, i], unique_gf, mode)
    merge_score_mat = merge_score_mat.numpy()
    merge_score_mat = np.nan_to_num(merge_score_mat)
    
    return score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids)

    
def collect_scoremat(model_list=['biggait', 'cal', 'adaface'], dataset='ccvid'):
    """
    Collect score matrices from different models and save them in a single h5 file.
    Args:
        model_list: list of model names
        dataset: dataset name
    Returns:
        None
    """
    f = h5py.File(osp.join(FILE_ROOT, 'test_feats', f'scoremats_{dataset}.h5'), 'w')
    meta_written = False

    for model_name in model_list:
        score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(osp.join(FILE_ROOT, 'test_feats', f'{model_name}_{dataset}_test.h5'), model_name)
        f.create_dataset(f'{model_name}/score_mat', data=score_mat)
        f.create_dataset(f'{model_name}/merge_score_mat', data=merge_score_mat)
        if not meta_written:
            f.create_dataset('q_pids', data=q_pids)
            f.create_dataset('q_camids', data=q_camids)
            f.create_dataset('q_clothes_ids', data=q_clothes_ids)
            f.create_dataset('g_pids', data=g_pids)
            f.create_dataset('g_camids', data=g_camids)
            f.create_dataset('g_clothes_ids', data=g_clothes_ids)
            f.create_dataset('unique_g_pids', data=unique_g_pids)
            meta_written = True
        f.flush()
    f.close()    


def merge_scoremats(model_list=['adaface', 'biggait', 'cal-ccvid'], dataset='ccvid', dataset_type='test', few_shot=None):
    """
    Merge score matrices from single score mat h5 files.
    Args:
        model_list: list of model names
        dataset: dataset name
    Returns:
        None
    """
    if dataset_type == 'test':
        h5_file_path = osp.join(FILE_ROOT, 'test_feats', f'scoremats_{dataset}.h5')
    else:
        if few_shot is None:
            h5_file_path = osp.join(FILE_ROOT, 'train_feats', f'scoremats_{dataset}_train.h5')
        else:
            h5_file_path = osp.join(FILE_ROOT, 'train_feats', f'scoremats_{dataset}_train_fewshot{few_shot}.h5')
    f = h5py.File(h5_file_path, 'w')
    for model_name in model_list:
        if dataset_type == 'test':
            f2 = h5py.File(osp.join(FILE_ROOT, 'test_feats', f'scoremat_{model_name}_{dataset}_test.h5'), 'r')
        else:
            if few_shot is None:
                f2 = h5py.File(osp.join(FILE_ROOT, 'train_feats', f'scoremat_{model_name}_{dataset}_train.h5'), 'r')
            else:
                f2 = h5py.File(osp.join(FILE_ROOT, 'train_feats', f'scoremat_{model_name}_{dataset}_train_fewshot{few_shot}.h5'), 'r')
        score_mat = f2['score_mat'][:]
        merge_score_mat = f2['merge_score_mat'][:]
        q_pids = f2['q_pids'][:]
        q_camids = f2['q_camids'][:]
        q_clothes_ids = f2['q_clothes_ids'][:]
        g_pids = f2['g_pids'][:]
        g_camids = f2['g_camids'][:]
        g_clothes_ids = f2['g_clothes_ids'][:]
        unique_g_pids = f2['unique_g_pids'][:]
        f2.close()
        f.create_dataset(f'{model_name}/score_mat', data=score_mat)
        f.create_dataset(f'{model_name}/merge_score_mat', data=merge_score_mat)
        # labels should be the same for all models
        if 'q_pids' in f:
            assert np.array_equal(f['q_pids'][:], q_pids), "Mismatch in q_pids"
            assert np.array_equal(f['q_camids'][:], q_camids), "Mismatch in q_camids"
            assert np.array_equal(f['q_clothes_ids'][:], q_clothes_ids), "Mismatch in q_clothes_ids"
            assert np.array_equal(f['g_pids'][:], g_pids), "Mismatch in g_pids"
            assert np.array_equal(f['g_camids'][:], g_camids), "Mismatch in g_camids"
            assert np.array_equal(f['g_clothes_ids'][:], g_clothes_ids), "Mismatch in g_clothes_ids"
            assert np.array_equal(f['unique_g_pids'][:], unique_g_pids), "Mismatch in unique_g_pids"
        else:
            f.create_dataset(f'q_pids', data=q_pids)
            f.create_dataset(f'q_camids', data=q_camids)
            f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
            f.create_dataset(f'g_pids', data=g_pids)
            f.create_dataset(f'g_camids', data=g_camids)
            f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
            f.create_dataset(f'unique_g_pids', data=unique_g_pids)
    f.close()


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    # parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', default='./WBModules/CAL/configs/res50_cels_cal.yaml')
    parser.add_argument('--mode', type=str, default='kprpe,adaface,cal-ccvid,biggait', help='model type/version name')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory", default='/localscratch/zhujie4/data/')
    parser.add_argument('--dataset', type=str, default='ccvid', help="ccvid, mevid", choices=['ccvid', 'mevid', 'ltcc'])
    parser.add_argument('--dataset_type', type=str, default='test', help='test or train', choices=['test', 'train'])
    parser.add_argument('--train_batch', type=int, default=1, help='batch size for extracting train features')
    parser.add_argument('--num_sample', type=int, default=4, help='number of samples for each query')
    parser.add_argument('--max_batch', type=int, default=-1, help='max batch size for extracting train features')
    parser.add_argument('--few_shot', type=int, default=None, help='few shot for extracting train features')
    parser.add_argument('--eval_mode', type=str, default='gather', choices=['score', 'feat', 'gather'], help='Evaluation type: [score, feat, gather]. feat: extract features for test set, score: compute score matrices for test set, gather: collect score matrices for all models')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29620'
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s : %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    config = parse_option()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    
    if config.eval_mode == 'feat':
        # /---------------Feature Inference Func-------------------/
        backbone_cfg = yaml.safe_load(open(osp.join(FILE_ROOT, 'WBModules', f'model_cfg_{config.DATA.DATASET}.yaml'), 'r'))
        backbone_cfg['model_list'] = [config.mode]
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(backbone_cfg['model_list'], config)
        model = QME(backbone_cfg)
        model = model.cuda()
        if config.dataset_type == 'test':
            fname = extract_test_feats(config, model, queryloader, galleryloader, dataset)
            f = h5py.File(fname, 'r')
            data_dict = {
                'qf': f['qf'][:], 
                'q_pids': f['q_pids'][:], 
                'q_camids': f['q_camids'][:], 
                'q_clothes_ids': f['q_clothes_ids'][:],
                'gf': f['gf'][:], 
                'g_pids': f['g_pids'][:], 
                'g_camids': f['g_camids'][:], 
                'g_clothes_ids': f['g_clothes_ids'][:]
            }
            f.close()
            score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(data_dict, mode=config.mode)
            res = test_score(score_mat, merge_score_mat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, config.DATA.DATASET, log=True, seed=45)
            # save score matrix
            f = h5py.File(osp.join(FILE_ROOT, 'test_feats', f'scoremat_{config.mode}_{config.DATA.DATASET}_test.h5'), 'w')
            f.create_dataset(f'score_mat', data=score_mat)
            f.create_dataset(f'merge_score_mat', data=merge_score_mat)
            f.create_dataset(f'q_pids', data=q_pids)
            f.create_dataset(f'q_camids', data=q_camids)
            f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
            f.create_dataset(f'g_pids', data=g_pids)
            f.create_dataset(f'g_camids', data=g_camids)
            f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
            f.create_dataset(f'unique_g_pids', data=unique_g_pids)
            f.flush()
            f.close()
        else:
            if config.max_batch > 0:
                qf, q_pids, q_camids, q_clothes_ids = extract_train_feats(config, model, trainloader, dataset, num_data=config.max_batch)
            else:
                qf, q_pids, q_camids, q_clothes_ids = extract_train_feats(config, model, trainloader, dataset)
            
            if config.few_shot is None:
                # load center features for dataset-level, as we need to align the center labels with other models.
                center_features = load_center_feat(model_list=DATASET_MAPPING[config.DATA.DATASET], dataset=dataset, device='cpu')
                # compute score matrix
                data_dict = {
                    'qf': qf, 
                    'q_pids': q_pids, 
                    'q_camids': q_camids, 
                    'q_clothes_ids': q_clothes_ids,
                    'gf': center_features['center_feats'][config.mode],
                    'g_pids': center_features['center_pids'],
                    'g_camids': center_features['center_camids'],
                    'g_clothes_ids': center_features['center_clothes_ids']
                }
                score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(data_dict, mode=config.mode)
                print(f"score mat shape: {score_mat.shape}, merge score mat shape: {merge_score_mat.shape}")
                # save score matrix
                f = h5py.File(osp.join(FILE_ROOT, 'train_feats', f'scoremat_{config.mode}_{config.DATA.DATASET}_train.h5'), 'w')
                f.create_dataset(f'score_mat', data=score_mat)
                f.create_dataset(f'merge_score_mat', data=merge_score_mat)
                f.create_dataset(f'q_pids', data=q_pids)
                f.create_dataset(f'q_camids', data=q_camids)
                f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
                f.create_dataset(f'g_pids', data=g_pids)
                f.create_dataset(f'g_camids', data=g_camids)
                f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
                f.create_dataset(f'unique_g_pids', data=unique_g_pids)
                f.flush()
                f.close()
            else:
                # for few shot setting, we need to create new center features based on the few shot samples.
                df = pd.DataFrame({
                    'pids': q_pids.numpy(),
                })
                qf_np = qf.numpy()

                grouped = df.groupby('pids')

                center_feats_list = []
                center_pids_list = []

                for pid, group in grouped:
                    indices = group.index
                    group_feats = qf_np[indices]
                    mean_feat = np.nanmean(group_feats, axis=0)
                    center_feats_list.append(mean_feat)
                    center_pids_list.append(pid)

                center_feats = torch.from_numpy(np.array(center_feats_list))
                center_pids = torch.from_numpy(np.array(center_pids_list, dtype=np.int64))
                
                num_centers = len(center_pids)
                # for compatibility with the code, we set the camids and clothes_ids to 9999 for further training.
                center_camids = torch.full((num_centers,), 9999, dtype=torch.int64)
                center_clothes_ids = torch.full((num_centers,), 9999, dtype=torch.int64)

                center_features = {
                    'center_feats': center_feats,
                    'center_pids': center_pids,
                    'center_camids': center_camids,
                    'center_clothes_ids': center_clothes_ids
                }
                # save center features
                save_path = f'./src/fusionagent/mod_center_feat/{config.mode}_center_{config.DATA.DATASET}_fewshot{config.few_shot}.h5'
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                with h5py.File(save_path, 'w') as f:
                    for i in range(len(center_pids)):
                        pid = center_pids[i].item()
                        clothes_id = center_clothes_ids[i].item()
                        camid = center_camids[i].item()
                        feat = center_feats[i].numpy()
                        f.create_dataset(f'{pid}/{camid}_{clothes_id}', data=feat[np.newaxis, :])
                # compute score matrix
                data_dict = {
                    'qf': qf, 
                    'q_pids': q_pids, 
                    'q_camids': q_camids, 
                    'q_clothes_ids': q_clothes_ids,
                    'gf': center_features['center_feats'],
                    'g_pids': center_features['center_pids'],
                    'g_camids': center_features['center_camids'],
                    'g_clothes_ids': center_features['center_clothes_ids']
                }
                score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(data_dict, mode=config.mode)
                print(f"score mat shape: {score_mat.shape}, merge score mat shape: {merge_score_mat.shape}")

                # save score matrix
                save_dir = f'./src/fusionagent/train_feats'
                os.makedirs(save_dir, exist_ok=True)
                f = h5py.File(f'{save_dir}/scoremat_{config.mode}_{config.DATA.DATASET}_train_fewshot{config.few_shot}.h5', 'w')
                f.create_dataset(f'score_mat', data=score_mat)
                f.create_dataset(f'merge_score_mat', data=merge_score_mat)
                f.create_dataset(f'q_pids', data=q_pids)
                f.create_dataset(f'q_camids', data=q_camids)
                f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
                f.create_dataset(f'g_pids', data=g_pids)
                f.create_dataset(f'g_camids', data=g_camids)
                f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
                f.create_dataset(f'unique_g_pids', data=unique_g_pids)
                f.flush()
                f.close()
    
    elif config.eval_mode == 'score':
        # /---------------Score Evaluate Func-------------------/
        saved_feat = osp.join(FILE_ROOT, 'test_feats', f'{config.mode}_{config.DATA.DATASET}_test.h5')
        f = h5py.File(saved_feat, 'r')
        data_dict = {
            'qf': f['qf'][:], 
            'q_pids': f['q_pids'][:], 
            'q_camids': f['q_camids'][:], 
            'q_clothes_ids': f['q_clothes_ids'][:],
            'gf': f['gf'][:], 
            'g_pids': f['g_pids'][:], 
            'g_camids': f['g_camids'][:], 
            'g_clothes_ids': f['g_clothes_ids'][:]
        }
        f.close()
        score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(data_dict, mode=config.mode)
        res = test_score(score_mat, merge_score_mat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, config.DATA.DATASET, log=True, seed=45)
        # save score matrix
        f = h5py.File(osp.join(FILE_ROOT, 'test_feats', f'scoremat_{config.mode}_{config.DATA.DATASET}_test.h5'), 'w')
        f.create_dataset(f'score_mat', data=score_mat)
        f.create_dataset(f'merge_score_mat', data=merge_score_mat)
        f.create_dataset(f'q_pids', data=q_pids)
        f.create_dataset(f'q_camids', data=q_camids)
        f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
        f.create_dataset(f'g_pids', data=g_pids)
        f.create_dataset(f'g_camids', data=g_camids)
        f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
        f.create_dataset(f'unique_g_pids', data=unique_g_pids)
        f.flush()
        f.close()
        
    elif config.eval_mode == 'gather':    
        # /---------------Merge Score Matrices-------------------/
        # os.chdir('./src/fusionagent')
        model_list = config.mode.split(',')
        if config.few_shot is None:
            merge_scoremats(model_list=model_list, dataset=config.DATA.DATASET, dataset_type=config.dataset_type)
        else:
            merge_scoremats(model_list=model_list, dataset=config.DATA.DATASET, dataset_type=config.dataset_type, few_shot=config.few_shot)
        