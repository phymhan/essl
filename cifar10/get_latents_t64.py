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

import pdb
st = pdb.set_trace

class ResNetWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return x

# CUDA_VISIBLE_DEVICES=3 python3 get_latents.py --batch_size 128 --n_part 6 --part 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    # parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='data_t64/t64_data.pkl')
    parser.add_argument('--latent_path', type=str, default='data_t64/t64_latent.pkl')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--n_part', type=int, default=None)
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--uint8', action='store_true')
    args = parser.parse_args()

    if args.merge:
        assert args.n_part is not None
        latents = []
        for part in range(args.n_part):
            # latent_path = args.latent_path.replace('.pkl', f'_{part}.pkl')
            latent_path = args.latent_path+f'.part{args.n_part}_{part}'
            with open(latent_path, 'rb') as f:
                latents.append(pickle.load(f)['latents'])
        latents = torch.cat(latents, dim=0)
        with open(args.latent_path, 'wb') as f:
            pickle.dump({'latents': latents}, f)
        exit(0)
    
    device = 'cuda'
    image_size = args.image_size
    # tol = 1e-3

    g_model_path = '/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0401_gan_t64/weight/latest.pt'
    g_ckpt = torch.load(g_model_path, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    e_model_path = '/research/cbim/medical/lh599/active/stylegan2-encoder-pytorch/checkpoint_t64/encoder_980000.pt'
    e_ckpt = torch.load(e_model_path, map_location=device)

    encoder = Encoder(image_size, latent_dim).to(device)
    encoder.load_state_dict(e_ckpt['e'])
    encoder.eval()
    print('[encoder loaded]')

    truncation = args.truncation
    trunc = generator.mean_latent(4096).detach().clone()
    batch_size = args.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.verify:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        images = data['images']
        with open(args.latent_path, 'rb') as f:
            latents = pickle.load(f)['latents']
        index = random.sample(range(len(latents)), 64)
        print(index)
        imgs_rec = []
        imgs_real = []
        for i in index:
            z = latents[i]
            img_rec, _ = generator([z.unsqueeze(0).to(device)], 
                                    input_is_latent=True, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            imgs_rec.append(img_rec.detach().cpu())
            imgs_real.append(images[i])
        imgs_rec = torch.stack(imgs_rec, dim=0).view(8, 8, 3, image_size, image_size)
        imgs_real = torch.stack(imgs_real, dim=0).view(8, 8, 3, image_size, image_size)
        img = []
        for i in range(8):
            img_row = []
            for j in range(8):
                img_row.append(torch.cat([imgs_real[i, j], imgs_rec[i, j]], dim=2))
            img_row = torch.cat(img_row, dim=2)
            img.append(img_row)
        img = torch.cat(img, dim=1)
        imsave('data_t64/verify.png', tensor2image(img))
        exit(0)
    
    if args.part is not None:
        assert args.n_part is not None
        seed = args.seed + args.part
        # args.latent_path = args.latent_path.replace('.pkl', f'_{args.part}.pkl')
        args.latent_path = args.latent_path+f'.part{args.n_part}_{args.part}'
    else:
        seed = args.seed
    utils.fix_seed(seed)

    data = {}
    if os.path.exists(args.data_path):
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        images = data['images']
        labels = data['labels']
        print('[data loaded]')
    else:
        counter = 0
        images = []
        labels = []
        print('caching data...')
        # images_ = []
        for mode in ['train', 'test']:
            # dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform, train=mode=='train')
            # dataset_ = datasets.CIFAR100(args.data_root, download=True, train=mode=='train')
            dataset_ = torchvision.datasets.ImageFolder(
                root=os.path.join('/research/cbim/medical/lh599/active/BigGAN/data/tiny-imagenet-200', 'train' if mode=='train' else 'val_imagefolder'),
                # transform=transform if not args.uint8 else None,
            )
            for i in tqdm(range(len(dataset_))):
                img_, lbl = dataset_[i]
                # img_ = dataset_[i][0]  # uint8 [64, 64, 3]
                img_ = np.array(img_)
                if args.uint8:
                    img = img_
                else:
                    img = transform(img_)
                images.append(img)
                labels.append(lbl)
                # images_.append(np.array(img_))
                counter += 1
        if args.uint8:
            images = np.stack(images)
        else:
            images = torch.stack(images)
        data['images'] = images
        data['labels'] = labels
        # data['uint8'] = images_
        print(f'[{counter} images cached]')
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        with open(args.data_path, 'wb') as f:
            pickle.dump(data, f)
    
    if args.uint8:  # saved images are uint8
        input_transform = transform
    else:
        input_transform = lambda x: x  # images are already float32

    vgg_loss = VGGLoss(device)

    toggle_grad(generator, False)
    toggle_grad(encoder, False)

    latents = []
    index = np.arange(len(data['labels']))
    if args.part is not None:
        part_size = int(np.ceil(len(index) / args.n_part))
        index = index[args.part * part_size:(args.part + 1) * part_size]

    for i in tqdm(range(int(np.ceil(len(index) / batch_size)))):
        inds = index[i * batch_size:(i + 1) * batch_size]
        imgs = images[inds]
        imgs = torch.stack([input_transform(x) for x in imgs])
        imgs = imgs.to(device)

        with torch.no_grad():
            z0 = encoder(imgs)

        z = z0.detach().clone()
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.lr_inv)
        for step in range(args.iters_inv):
            imgs_gen, _ = generator([z], 
                                    input_is_latent=True, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            z_hat = encoder(imgs_gen)
            loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + F.mse_loss(z0, z_hat)*2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        z1 = z.detach().cpu().clone()
        z1.requires_grad = False
        latents.append(z1)
    latents = torch.cat(latents, dim=0)
    with open(args.latent_path, 'wb') as f:
        pickle.dump({'latents': latents}, f)