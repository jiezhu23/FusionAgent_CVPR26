import torch
import functools
import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, is_training=False):
        self.dataset = dataset
        self.transform = transform
        self.is_training = is_training
        if 'jpg' in self.dataset['dataset'][0][0] or 'png' in self.dataset['dataset'][0][0]:
            self.dataset_type = 'img'
        else:
            self.dataset_type = 'h5'
        # lazy import to avoid circular import
        from WBModules import MODEL_MAPPING_DICT
        self.model_mapping_dict = MODEL_MAPPING_DICT

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset['dataset'][index]
        # get data based on key value of transform
        clips = {}
        for mode in self.transform.keys():
            if self.dataset_type == 'img':
                img = read_image(img_path)
                clip = [img]
            else:
                if self.model_mapping_dict[mode] == 'body_data':
                    clip = [Image.fromarray(self.dataset[self.model_mapping_dict[mode]][img_path][:])]
                else:
                    clip = self.face_sampler(img_path)
            if self.transform is not None:
                clip = [self.transform[mode](img) for img in clip]
            # trans (C x H x W) to (C x T=1 x H x W)
            if len(clip) == 0:
                clip = torch.zeros((0, 0, 0, 0))
            else:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips[mode] = clip

        return clips, pid, camid, clothes_id

    def face_sampler(self, img_path):
        clip = []
        if img_path in self.dataset['face_data']:
            face_idx = list(self.dataset['face_data'][img_path].keys())
            clip.append(Image.fromarray(self.dataset['face_data'][img_path][random.choice(face_idx)][:]))
        if self.is_training and len(clip) < 1:
            # Random padding from whole tracklet/video to make the length of face data equal to img_paths
            if self.dataset['dataset_name'] == 'ltcc':
                folder, img_name = osp.split(img_path)
                id, clothes_id, cam_id, _  = img_name.split('_')
                # randomly sample facial images from the same subject
                random_f = random.choice(self.dataset['id2face'][id])
                face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
        return clip


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (dict):
                Key: 'dataset' (list) - List with items (img_paths, pid, camid)
                Key: 'body_data' (h5 file pointer) - h5 file pointer to body data
                Key: 'face_data' (h5 file pointer) - h5 file pointer to face data
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True,
                 is_training=False):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader =get_loader()
        self.cloth_changing = cloth_changing
        self.is_training = is_training
        if 'jpg' in self.dataset['dataset'][0][0][0]:
            self.dataset_type = 'img'
        else:
            self.dataset_type = 'h5'
        # lazy import to avoid circular import
        from WBModules import MODEL_MAPPING_DICT
        self.model_mapping_dict = MODEL_MAPPING_DICT

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset['dataset'][index]
        else:
            img_paths, pid, camid = self.dataset['dataset'][index]

        # get data based on key value of transform
        clips = {}
        for mode in self.spatial_transform.keys():
            if self.temporal_transform[mode] is not None:
                mode_img_paths = self.temporal_transform[mode](img_paths)
            else:
                mode_img_paths = img_paths
            if self.dataset_type == 'img':
                clip = self.loader(mode_img_paths)
            else:
                if self.model_mapping_dict[mode] == 'body_data':
                    clip = [Image.fromarray(self.dataset[self.model_mapping_dict[mode]][p][:]) for p in mode_img_paths]
                else:
                    clip = self.face_sampler(mode_img_paths)

            if self.spatial_transform[mode] is not None:
                if 'qwen2.5' in mode:
                    clip = self.spatial_transform[mode](videos=clip, return_tensors='pt')['pixel_values']
                else:
                    if hasattr(self.spatial_transform[mode], 'randomize_parameters'):
                        self.spatial_transform[mode].randomize_parameters()
                    clip = [self.spatial_transform[mode](img) for img in clip]

            # trans T x C x H x W to C x T x H x W
            if len(clip) == 0:
                if self.is_training:
                    # create a zero tensor if face data is not available for subjects
                    clip = torch.zeros((3, len(mode_img_paths), 112, 112))
                else:
                    clip = torch.zeros((0, 0, 0, 0))
            else:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips[mode] = clip
            
        if self.cloth_changing:
            return clips, pid, camid, clothes_id
        else:
            return clips, pid, camid
        
    def face_sampler(self, img_paths):
        clip = []
        for p in img_paths:
            if p in self.dataset['face_data']:
                # one img may have multiple faces, sample one face randomly
                face_keys = list(self.dataset['face_data'][p].keys())
                clip.append(Image.fromarray(self.dataset['face_data'][p][random.choice(face_keys)][:]))
        if self.is_training:
            # Random padding from whole tracklet/video to make the length of face data equal to img_paths
            while len(clip) < len(img_paths):
                if self.dataset['dataset_name'] == 'ccvid':
                    folder, img_name = osp.split(img_paths[0])
                    if folder not in self.dataset['face_data']:
                        # some video may not have any face detected, randomly sample from other training videos with the same pid
                        session, vid = folder.split('/')
                        pid_tmp, _ = vid.split('_')
                        vid_pools = [vid for vid in self.dataset['face_data'][session].keys() if vid.split('_')[0] == pid_tmp]
                        folder = osp.join(session,random.choice(vid_pools))
                    random_f = random.choice(list(self.dataset['face_data'][folder].keys()))
                    face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                    clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
                elif self.dataset['dataset_name'] == 'mevid':
                    folder, img_name = osp.split(img_paths[0])
                    if folder not in self.dataset['face_data']:
                        # this subejct has no face detected, return an empty list
                        return clip
                    random_f = random.choice(list(self.dataset['face_data'][folder].keys()))
                    face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                    clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
        return clip


class InterleaveDataset(Dataset):
    """
    Dataset class for MLLM finetuning that returns structured data format
    with img_path, pil_image, subject_id, and face_bbox information.
    
    Args:
        dataset (dict): Dataset dictionary containing 'dataset', 'body_data', 'face_data' keys
        config: Configuration object containing dataset information
        max_samples (int, optional): Maximum number of samples to include. If None, includes all samples.
    """
    
    def __init__(self, dataset, config, cloth_changing=True, max_samples=None, shuffle=False):
        self.dataset = dataset
        self.config = config
        self.cloth_changing = cloth_changing  # Assume cloth changing dataset by default
        self.max_samples = max_samples
        self.dataset['dataset_name'] = config.DATA.DATASET
        if shuffle:
            random.shuffle(self.dataset['dataset'])
            
        # if self.dataset['dataset_name'] == 'mevid':
        #     self._build_mevid_face_cache()
            
    def _build_interleaved_data(self):
        """Build the interleaved dataset from the original dataset structure"""
        rows = []
        
        for i, data_item in enumerate(self.dataset['dataset']):
            if self.cloth_changing:
                clip_paths, pid, camid, clothes_id = data_item
            else:
                clip_paths, pid, camid = data_item
                
            for img_path in clip_paths:
                img_dir = os.path.dirname(img_path)
                img_id = os.path.basename(img_path)
                
                # Check if face data exists for this image
                has_face_bbox = (img_dir in self.dataset.get('face_data', {}) and 
                               img_id in self.dataset['face_data'][img_dir])
                
                # Create subject ID
                subject_id = "_".join([self.config.DATA.DATASET, str(pid)])
                
                row = {
                    'img_path': img_path,
                    'subject_id': subject_id,
                    'has_face': has_face_bbox
                }
                rows.append(row)
                
                # Early termination if max_samples is set
                if self.max_samples is not None and len(rows) >= self.max_samples:
                    return rows
                    
        return rows

    def _build_mevid_face_cache(self):
        """Pre-computes a cache for faster face lookups for the MEVID dataset."""
        self.face_cache = {}
        face_data = self.dataset.get('face_data', {})
        if not face_data:
            return

        for img_dir, images_with_faces in face_data.items():
            face_paths = []
            for img_id, faces in images_with_faces.items():
                # Assuming img_id is a filename and faces is a dict of face crops for that image
                for face_key in faces.keys():
                    # The original code at L353 suggests the path is constructed like this
                    full_img_path = osp.join(img_dir, img_id)
                    face_paths.append(f'{full_img_path}/{face_key}')
            if face_paths:
                self.face_cache[img_dir] = face_paths
    
    def __len__(self):
        total = len(self.dataset['dataset'])
        if self.max_samples is None:
            return total
        try:
            max_n = int(self.max_samples)
        except Exception:
            return total
        if max_n <= 0:
            return total
        return min(total, max_n)
    
    def __getstate__(self):
        """
        Prepare the object for pickling.
        Replaces h5py file handles with their file paths to make it picklable.
        """
        state = self.__dict__.copy()
        # Create a picklable version of the 'dataset' dictionary
        picklable_dataset = {}
        for key, value in self.dataset.items():
            # Check if the object is an h5py File or Group
            if hasattr(value, 'filename'):
                picklable_dataset[key] = {'__h5py_file__': value.filename}
            else:
                picklable_dataset[key] = value
        state['dataset'] = picklable_dataset
        return state

    def __setstate__(self, state):
        """
        Restore the object after unpickling.
        Re-opens h5py file handles from their paths.
        """
        import h5py
        # Restore the 'dataset' dictionary
        unpickled_dataset = {}
        for key, value in state['dataset'].items():
            if isinstance(value, dict) and '__h5py_file__' in value:
                # Re-open the h5py file in read mode
                unpickled_dataset[key] = h5py.File(value['__h5py_file__'], 'r')
            else:
                unpickled_dataset[key] = value
        state['dataset'] = unpickled_dataset
        self.__dict__.update(state)
    
    def __getitem__(self, index):
        """
        Returns:
            dict: Dictionary containing:
                - img_path (str): Path to the image
                - pil_image (PIL.Image): PIL Image object
                - subject_id (str): Subject identifier in format "dataset_pid"
                - face_bbox (bool): Whether face bbox data exists for this image
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range for InterleaveDataset")
        if self.cloth_changing:
            clip_paths, pid, camid, clothes_id = self.dataset['dataset'][index]
        else:
            clip_paths, pid, camid = self.dataset['dataset'][index]
        
        if isinstance(clip_paths, str):
            clip_paths = [clip_paths]
        body_image_keys = []
        face_image_keys = []
        has_face_list = []
        for img_path in clip_paths:
            img_dir = os.path.dirname(img_path)
            img_id = os.path.basename(img_path)
            
            # Check if face data exists for this image
            has_face = (img_dir in self.dataset.get('face_data', {}) and 
                           img_id in self.dataset['face_data'][img_dir])
            
            # Create subject ID
            subject_id = "_".join([self.config.DATA.DATASET, str(pid)])
            if has_face:
                # mevid applied a data cleaning process, so the index may not be face_0
                face_keys = list(self.dataset['face_data'][img_path].keys())
                face_image_keys.append(img_path+f'/{face_keys[0]}')
            # else:
            #     # continue
            #     if self.dataset['dataset_name'] == 'mevid' and self.max_samples is not None:
            #         # find the face from the same subject for mevid
            #         face_dir = list(self.dataset['face_data'][img_dir].keys())
            #         if len(face_dir) > 0:
            #             selected_face_dir = random.choice(face_dir)
            #             face_keys = list(self.dataset['face_data'][img_dir][selected_face_dir].keys())
            #             face_image_keys.append(osp.join(img_dir, selected_face_dir, random.choice(face_keys)))
            #             has_face = True

            body_image_keys.append(img_path)
            has_face_list.append(has_face)
        sample = {
            'body_image_keys': body_image_keys,
            'face_image_keys': face_image_keys,
            'subject_id': subject_id,
            'has_face': has_face_list,
            'index': index,
        }

        # load image later    
        # for img_path in clip_paths:
        #     img_dir = os.path.dirname(img_path)
        #     img_id = os.path.basename(img_path)
            
        #     # Check if face data exists for this image
        #     has_face_bbox = (img_dir in self.dataset.get('face_data', {}) and 
        #                     img_id in self.dataset['face_data'][img_dir])
            
        #     # Create subject ID
        #     subject_id = "_".join([self.config.DATA.DATASET, str(pid)])
        #     # Create PIL image from body data
        #     pil_image = Image.fromarray(np.array(self.dataset['body_data'][img_path]))
        #     if has_face_bbox:
        #         # always select the first face
        #         face_pil_image = Image.fromarray(np.array(self.dataset['face_data'][img_path]['face_0']))
        #     else:
        #         face_pil_image = None
        #     row = {
        #         'img_path': img_path,
        #         'body_pil_image': pil_image,
        #         'face_pil_image': face_pil_image,
        #         'subject_id': subject_id,
        #         'has_face': has_face_bbox
        #     }
        #     rows.append(row)

        return sample

    def __iter__(self):
        """Yields data samples one by one."""
        for i in range(len(self)):
            yield self[i]