import os
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm

import pdb
st = pdb.set_trace

def convert_to_images(obj, min_val=-1):
    """ Convert an output tensor from BigGAN in a list of images.
    """
    # need to fix import, see: https://github.com/huggingface/pytorch-pretrained-BigGAN/pull/14/commits/68a7446951f0b9400ebc7baf466ccc48cdf1b14c
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 255, 0, 255)
    # obj = np.clip(obj * 256, 0, 255)
    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(Image.fromarray(out_array))
    return img

save_dir = '../../../data/CIFAR/cifar10-imagefolder/'

with open('data/c10_data.pkl', 'rb') as f:
    data = pickle.load(f)

images = data['images']  # (-1, 1)
labels = data['labels']

images = convert_to_images(images)
cnt = 0
for j, im in tqdm(enumerate(images)):
    cnt += 1
    classname = f"{labels[j]:02d}"
    imagename = f"{cnt:04d}.png"
    subset = 'train' if j < 50000 else 'test'
    os.makedirs(os.path.join(save_dir, subset, classname), exist_ok=True)
    im.save(os.path.join(
        save_dir, subset, classname, imagename))


save_dir = '../../../data/CIFAR/cifar100-imagefolder/'

with open('data_c100/c100_data.pkl', 'rb') as f:
    data = pickle.load(f)

images = data['images']  # (-1, 1)
labels = data['labels']

images = convert_to_images(images)
cnt = 0
for j, im in tqdm(enumerate(images)):
    cnt += 1
    classname = f"{labels[j]:02d}"
    imagename = f"{cnt:04d}.png"
    subset = 'train' if j < 50000 else 'test'
    os.makedirs(os.path.join(save_dir, subset, classname), exist_ok=True)
    im.save(os.path.join(
        save_dir, subset, classname, imagename))