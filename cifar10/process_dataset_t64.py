import os
import pickle
import shutil
from tqdm import tqdm


"""
https://github.com/jcjohnson/tiny-imagenet/blob/master/make_tiny_imagenet.py
"""

img_dir = '/research/cbim/medical/lh599/active/BigGAN/data/tiny-imagenet-200/val/images'
annotations = '/research/cbim/medical/lh599/active/BigGAN/data/tiny-imagenet-200/val/val_annotations.txt'

save_dir = '/research/cbim/medical/lh599/active/BigGAN/data/tiny-imagenet-200/val_imagefolder'

with open(annotations, 'r') as f:
    annotations = f.readlines()

for a in tqdm(annotations):
    x = a.split('\t')[0]
    y = a.split('\t')[1]
    if not os.path.exists(os.path.join(save_dir, y)):
        os.makedirs(os.path.join(save_dir, y))
    shutil.copyfile(os.path.join(img_dir, x), os.path.join(save_dir, y, x))
