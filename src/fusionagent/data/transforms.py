import numpy as np
from PIL import Image
import torch

import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT

class BaseRgbCuttingTransform:
    def __init__(self, mean=None, std=None, cutting=None, img_w=64):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

        self.img_w = img_w
        self.cutting = cutting

    def __call__(self, x):
        """
        x: (C, H, W) or (N, C, H, W)
        output: (C, H, W) or (N, C, H, W)
        """
        if self.cutting is not None:
            cutting = self.cutting
        else:
            if x.shape[-1] == x.shape[-2]:
                cutting = x.shape[-1]//4
            elif x.shape[-1] * 2 == x.shape[-2]:
                cutting = 0
            else:
                raise ValueError
        if cutting != 0:
            x = x[..., cutting:-cutting]
        else:
            x = x
        if x.ndim == 3:
            return ((x - self.mean) / self.std).squeeze(0) # (C, H, W)
        else:
            return (x - self.mean) / self.std # (N, C, H, W)


class BasePILCuttingTransform:
    def __init__(self, mean=None, std=None, cutting=None, img_w=64):
        if mean is None:
            mean = [0.485 , 0.456 , 0.406 ]
        if std is None:
            std = [0.229 , 0.224 , 0.225 ]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

        self.img_w = img_w
        self.cutting = cutting

    def __call__(self, x):
        """
        x: (C, H, W) or (N, C, H, W)
        output: (C, H, W) or (N, C, H, W)
        """
        # If input is a PIL image, convert to NumPy array
        if isinstance(x, Image.Image):
            x = np.array(x).astype(np.float32) / 255.0  # Convert to NumPy array and normalize to [0, 1]
            x = np.transpose(x, (2, 0, 1))  # Convert (H, W, C) to (C, H, W)
            
        if self.cutting is not None:
            cutting = self.cutting
        else:
            if x.shape[-1] == x.shape[-2]:
                cutting = x.shape[-1]//4
            elif x.shape[-1] * 2 == x.shape[-2]:
                cutting = 0
            else:
                raise ValueError
        if cutting != 0:
            x = x[..., cutting:-cutting]
        else:
            x = x
        if x.ndim == 3:
            return torch.tensor(((x - self.mean) / self.std).squeeze(0)).float() # (C, H, W)
        else:
            return torch.tensor((x - self.mean) / self.std).float() # (N, C, H, W)


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_vid_transforms(config):
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.HEIGHT, width=config.DATA.WIDTH, probability=config.AUG.RE_PROB)
    ])
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                         stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test


def get_transforms(mode='kprpe', config=None, dataset_type='vid'):
    # Define train and test transforms based on the mode
    if 'kprpe' in mode or 'adaface' in mode or 'arcface' in mode:
        # For face feature
        train_transform = T.Compose([
            T.Resize((112, 112)),
            T.BlurAugmenter(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if dataset_type == 'vid':
            if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
                temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
            elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
                temporal_transform_train = TT.TemporalRandomCrop(size=config.DATA.SAMPLING_STEP, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
            else:
                raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        else:
            temporal_transform_train = None
            
        temporal_transform_test = None
        
    elif 'vit' in mode or 'evl' in mode:
        # For body feature
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomCroping(p=0.2),
            T.ToTensor(),
            T.RandomErasing(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        temporal_transform_train, temporal_transform_test = None, None

    elif 'biggait' in mode:
        train_transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomCroping(p=0.2),
            T.ToTensor(),
            BasePILCuttingTransform(cutting=0),
            T.RandomErasing()
        ])
        test_transform = T.Compose([T.Resize((256, 128)),
                                    BasePILCuttingTransform(cutting=0)])
        if dataset_type == 'vid':
            if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
                temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
            elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
                temporal_transform_train = TT.TemporalRandomCrop(size=config.DATA.SAMPLING_STEP, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
            else:
                raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        else:
            temporal_transform_train = None
            
        temporal_transform_test = None

    elif 'cal' in mode:
        # override the height and width for pre-trained model
        if 'mevid' in mode:
            config.defrost()
            config.DATA.HEIGHT=256
            config.DATA.WIDTH=128
            config.AUG.SEQ_LEN=config.DATA.SAMPLING_STEP
            config.freeze()
        else:
            config.defrost()
            config.DATA.HEIGHT=384
            config.DATA.WIDTH=192
            if dataset_type == 'vid':
                config.AUG.SEQ_LEN=config.DATA.SAMPLING_STEP
            config.freeze()
        if dataset_type == 'img':
            train_transform, test_transform = build_img_transforms(config)
            temporal_transform_train, temporal_transform_test = None, None
        else:
            train_transform, test_transform, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)
    
    elif 'agrl' in mode:
        # from WBModules.AGRL.torchreid import transforms as AT
        train_transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
            temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
        elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
            temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
        else:
            raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        temporal_transform_test = None

    elif 'aim' in mode:
        config.defrost()
        config.DATA.HEIGHT=384
        config.DATA.WIDTH=192
        if dataset_type == 'img':
            train_transform, test_transform = build_img_transforms(config)
            temporal_transform_train, temporal_transform_test = None, None
        else:
            train_transform, test_transform, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)
    
    elif 'qwen' in mode:
        from transformers import AutoProcessor
        processing_class = AutoProcessor.from_pretrained(mode)
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        train_transform = None
        test_transform = processing_class
        temporal_transform_train, temporal_transform_test = None, None
    
    else:
        raise ValueError('Not a correct mode')

    return train_transform, test_transform, temporal_transform_train, temporal_transform_test

