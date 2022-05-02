import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torchvision import datasets, transforms

from stylegan_model import Generator, Encoder
from view_generator import VGGLoss

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import utils
from tqdm import tqdm
from utils import toggle_grad, image2tensor, tensor2image, imshow, imsave, Config
import pickle
from einops import rearrange
import pdb
st = pdb.set_trace

"""
pregenerate augmentation for simclr baseline
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=100)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--data_path', type=str, default='data/c10_data.pkl')
    parser.add_argument('--dest_data_path', type=str, default='data/c10_expert.pkl')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=0.5)
    parser.add_argument('--init_noise_scale', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--lam2', type=float, default=0, help='weight of distance regularization')
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--objective', type=str, default='norm', choices=['norm', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l2', 'l1', 'hinge'])
    parser.add_argument('--method', type=str, default='gd', choices=['gd', 'fgsm', 'vat'])
    parser.add_argument('--n_part', type=int, default=None)
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--expert_augs', type=str, default='colorjitter,grayscale')
    args = parser.parse_args()

    args.expert_augs = args.expert_augs.split(',')
    print(args.expert_augs)
    train_transform = []
    if 'resizedcrop' in args.expert_augs:
        train_transform.append(T.RandomResizedCrop(size=32, scale=(0.2, 1.0)))
    if 'horizontalflip' in args.expert_augs:
        train_transform.append(T.RandomHorizontalFlip(p=0.5))
    if 'colorjitter' in args.expert_augs:
        train_transform.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    if 'grayscale' in args.expert_augs:
        train_transform.append(T.RandomGrayscale(p=0.2))
    train_transform = T.Compose(train_transform)

    args.clamp = True
    device = 'cuda'
    image_size = args.image_size
    batch_size = args.batch_size
    utils.fix_seed(args.seed)
    n = args.n

    # load data
    assert os.path.exists(args.data_path)
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    images = data['images']
    labels = data['labels']
    print('[data loaded]')

    views_image = []

    index = np.arange(len(data['labels']))
    if args.part is not None:
        part_size = int(np.ceil(len(index) / args.n_part))
        index = index[args.part * part_size:(args.part + 1) * part_size]

    for i in tqdm(range(int(np.ceil(len(index) / batch_size)))):
        inds = index[i * batch_size:(i + 1) * batch_size]
        bsz = len(inds)
        imgs = images[inds].clone()
        imgs_aug = [train_transform(imgs) for _ in range(n)]
        imgs_aug = torch.stack(imgs_aug, dim=1)
        views_image.append(imgs_aug)

    views_image = torch.cat(views_image, dim=0)
    views = {'n_views': n}
    views['views'] = views_image.transpose(0, 1)  # [n_views, B, 3, H, W]
    with open(os.path.join(args.dest_data_path), 'wb') as f:
        pickle.dump(views, f)
