
from torch.utils.data import DataLoader
import numpy as np

import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from data.dataloader import DataLoaderX
from data.transforms import BaseRgbCuttingTransform, BasePILCuttingTransform, build_img_transforms, build_vid_transforms, get_transforms
from data.dataset_loader import ImageDataset, VideoDataset, InterleaveDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.ccvid import CCVID
from data.datasets.mevid import MEVID
# from datasets.last import LaST
# from datasets.deepchange import DeepChange
# from datasets.vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    # 'vcclothes': VCClothes,
    # 'vcclothes_sc': VCClothesSameClothes,
    # 'vcclothes_cc': VCClothesClothesChanging,
    # 'last': LaST,
    'ccvid': CCVID,
    'mevid': MEVID,
    # 'deepchange': DeepChange,
}
# register new video datasets here
VID_DATASET = ['ccvid', 'mevid']


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    if config.DATA.DATASET in VID_DATASET:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, 
                                                 sampling_step=config.DATA.SAMPLING_STEP,
                                                 seq_len=config.AUG.SEQ_LEN, 
                                                 stride=config.AUG.SAMPLING_STRIDE,
                                                 few_shot=config.few_shot if hasattr(config, 'few_shot') else None)
    else:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, few_shot=config.few_shot if hasattr(config, 'few_shot') else None)
    dataset.dataset_name = config.DATA.DATASET
    return dataset


def build_dataloader(mode, config, is_training=True):
    dataset = build_dataset(config)
    # video dataset
    if config.DATA.DATASET in VID_DATASET:
        
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = {}, {}, {}, {}
        for m in mode:
            spatial_transform_train_m, spatial_transform_test_m, temporal_transform_train_m, temporal_transform_test_m = get_transforms(m, config, dataset_type='vid')
            spatial_transform_train[m] = spatial_transform_train_m
            spatial_transform_test[m] = spatial_transform_test_m
            temporal_transform_train[m] = temporal_transform_train_m
            temporal_transform_test[m] = temporal_transform_test_m
        
        # spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)

        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train, is_training=is_training),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_train, temporal_transform_train, is_training=is_training),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        
        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test, is_training=False),
            sampler=DistributedInferenceSampler(dataset.recombined_query['dataset']),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test, is_training=False),
            sampler=DistributedInferenceSampler(dataset.recombined_gallery['dataset']),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler
    # image dataset
    else:
        transform_train, transform_test = {}, {}
        for m in mode:
            transform_train_m, transform_test_m, _, _ = get_transforms(m, config, dataset_type='img')
            transform_train[m] = transform_train_m
            transform_test[m] = transform_test_m
        # transform_train, transform_test = build_img_transforms(config)
        train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                         num_instances=config.DATA.NUM_INSTANCES, 
                                                         seed=config.SEED)
        trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train, is_training=is_training),
                                 sampler=train_sampler,
                                 batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=True)

        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test, is_training=False),
                                   sampler=DistributedInferenceSampler(dataset.gallery['dataset']),
                                   batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=True, drop_last=False, shuffle=False)

        if config.DATA.DATASET == 'prcc':
            queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query_same),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)
            queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query_diff),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
        else:
            queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query['dataset']),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader, galleryloader, dataset, train_sampler


def build_agent_dataset(mode, config, is_training=True):
    dataset = build_dataset(config)
    # video dataset
    if config.DATA.DATASET in VID_DATASET:
        
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = {}, {}, {}, {}
        for m in mode:
            spatial_transform_train_m, spatial_transform_test_m, temporal_transform_train_m, temporal_transform_test_m = get_transforms(m, config, dataset_type='vid')
            spatial_transform_train[m] = spatial_transform_train_m
            spatial_transform_test[m] = spatial_transform_test_m
            temporal_transform_train[m] = temporal_transform_train_m
            temporal_transform_test[m] = temporal_transform_test_m
        
        # spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)

        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            train_dataset = VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train, is_training=is_training)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            train_dataset = VideoDataset(dataset.train, spatial_transform_train, temporal_transform_train, is_training=is_training)
        
        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        query_dataset = VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test, is_training=False)
        gallery_dataset = VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test, is_training=False)

        return train_dataset, query_dataset, gallery_dataset, dataset, train_sampler
    
    # image dataset
    else:
        transform_train, transform_test = {}, {}
        for m in mode:
            transform_train_m, transform_test_m, _, _ = get_transforms(m, config, dataset_type='img')
            transform_train[m] = transform_train_m
            transform_test[m] = transform_test_m
        # transform_train, transform_test = build_img_transforms(config)
        train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                         num_instances=config.DATA.NUM_INSTANCES, 
                                                         seed=config.SEED)
        train_dataset = ImageDataset(dataset.train, transform=transform_train, is_training=is_training)

        gallery_dataset = ImageDataset(dataset.gallery, transform=transform_test, is_training=False)

        if config.DATA.DATASET == 'prcc':
            query_same_dataset = ImageDataset(dataset.query_same, transform=transform_test, is_training=False)
            query_diff_dataset = ImageDataset(dataset.query_diff, transform=transform_test, is_training=False)

            return train_dataset, query_same_dataset, query_diff_dataset, gallery_dataset, dataset, train_sampler
        else:
            query_dataset = ImageDataset(dataset.query, transform=transform_test, is_training=False)


            return train_dataset, query_dataset, gallery_dataset, dataset, train_sampler


def build_interleave_dataset(config, max_samples=None):
    """
    Build InterleaveDataset for MLLM finetuning
    
    Args:
        mode (list): List of model modes (e.g., ['cal-ccvid'])
        config: Configuration object
        max_samples (int, optional): Maximum number of samples to include
        
    Returns:
        InterleaveDataset: Dataset instance for MLLM finetuning
    """
    # Get the raw dataset
    dataset = build_dataset(config)
    
    # Use the training data for interleaving
    if config.DATA.DATASET in VID_DATASET:
        if config.DATA.DENSE_SAMPLING:
            train_data_container = dataset.train_dense
        else:
            train_data_container = dataset.train
        
        # Subsample here before creating InterleaveDataset to avoid loading all data
        if max_samples is not None and len(train_data_container['dataset']) > max_samples:
            indices = np.random.choice(len(train_data_container['dataset']), max_samples, replace=False)
            train_data_container['dataset'] = [train_data_container['dataset'][i] for i in indices]

        if config.DATA.DENSE_SAMPLING:
            train_dataset = InterleaveDataset(train_data_container, config, max_samples=max_samples, shuffle=True)
        else:
            train_dataset = InterleaveDataset(train_data_container, config, max_samples=max_samples, shuffle=True)

        # for evaluation, we do not use recombined query
        # We use the original query and select fixed frames for agent to decide the model combination
        query_dataset = InterleaveDataset(dataset.query, config, max_samples=None)
        gallery_dataset = InterleaveDataset(dataset.gallery, config, max_samples=None)
        
        return train_dataset, query_dataset, gallery_dataset, dataset
    
    else:
        train_dataset = InterleaveDataset(dataset.train, config, max_samples=max_samples, shuffle=True)
        gallery_dataset = InterleaveDataset(dataset.gallery, config, max_samples=None)
        if config.DATA.DATASET == 'prcc':
            query_same_dataset = InterleaveDataset(dataset.query_same, config, max_samples=None)
            query_diff_dataset = InterleaveDataset(dataset.query_diff, config, max_samples=None)
            return train_dataset, query_same_dataset, query_diff_dataset, gallery_dataset, dataset
        else:
            query_dataset = InterleaveDataset(dataset.query, config, max_samples=None)
            return train_dataset, query_dataset, gallery_dataset, dataset

