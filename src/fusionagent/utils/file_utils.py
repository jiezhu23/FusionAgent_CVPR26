import os
import sys
import shutil
import errno
import json
import os.path as osp
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import h5py
# from deepface import DeepFace
# from retinaface import RetinaFace


def visualize_face_bbox(img_path):
    try:
        cropped_faces = DeepFace.extract_faces(img_path, detector_backend = 'retinaface', align=False)
    except ValueError:
        cropped_faces = []

    img = cv2.imread(img_path)

    if len(cropped_faces) == 0:
        print('No face detected in the image')
    else:
        for face in cropped_faces:
            # Extract the facial area coordinates
            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']

            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()
    

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def load_scoremats_from_h5(h5_file_path, model_list):
    f = h5py.File(h5_file_path, 'r')
    score_mats, merge_score_mats = {}, {}
    q_pids = f['q_pids'][:].astype(int)
    q_camids = f['q_camids'][:].astype(int)
    q_clothes_ids = f['q_clothes_ids'][:].astype(int)
    g_pids = f['g_pids'][:].astype(int)
    g_camids = f['g_camids'][:].astype(int)
    g_clothes_ids = f['g_clothes_ids'][:].astype(int)
    unique_g_pids = f['unique_g_pids'][:].astype(int)
    for model_name in model_list:
        score_mats[model_name] = torch.from_numpy(f[model_name]['score_mat'][:])
        merge_score_mats[model_name] = torch.from_numpy(f[model_name]['merge_score_mat'][:])
    dict_feats = {
        'q_pids': q_pids,
        'g_pids': g_pids,
        'q_camids': q_camids,
        'g_camids': g_camids,
        'q_clothes_ids': q_clothes_ids,
        'g_clothes_ids': g_clothes_ids,
        'unique_g_pids': unique_g_pids,
        'score_mats': score_mats,
        'merge_score_mats': merge_score_mats,
    }
    return dict_feats


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
