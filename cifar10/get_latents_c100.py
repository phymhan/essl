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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='data_c100/c100_data.pkl')
    parser.add_argument('--latent_path', type=str, default='data_c100/c100_latent.pkl')
    parser.add_argument('--n_part', type=int, default=None)
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--g_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0423_gan_c100/weight/220000.pt')
    parser.add_argument('--e_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-encoder-pytorch/checkpoint_c100/encoder_980000.pt')
    parser.add_argument('--n_latent', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--uniform_noise', action='store_true')
    args = parser.parse_args()

    if args.merge:
        assert args.n_part is not None
        latents = []
        for part in range(args.n_part):
            # latent_path = args.latent_path.replace('.pkl', f'_{part}.pkl')
            latent_path = args.latent_path+f'.part{part}'
            with open(latent_path, 'rb') as f:
                latents.append(pickle.load(f)['latents'])
        latents = torch.cat(latents, dim=0)
        with open(args.latent_path, 'wb') as f:
            pickle.dump({'latents': latents}, f)
        exit(0)
    
    device = 'cuda'
    image_size = args.image_size
    # tol = 1e-3

    # g_model_path = '/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0423_gan_c100/weight/220000.pt'
    g_ckpt = torch.load(args.g_ckpt, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    # e_model_path = '/research/cbim/medical/lh599/active/stylegan2-encoder-pytorch/checkpoint_c100/encoder_980000.pt'
    if args.e_ckpt is not None and os.path.exists(args.e_ckpt):
        e_ckpt = torch.load(args.e_ckpt, map_location=device)
        encoder = Encoder(image_size, latent_dim).to(device)
        encoder.load_state_dict(e_ckpt['e'])
        encoder.eval()
        print('[encoder loaded]')
    else:
        encoder = None
        print('[no encoder]')
    
    assert latent_dim == args.latent_dim
    if args.uniform_noise:
        truncation = 1
        trunc = None
        input_is_latent = False
        n_latent = 1
        latent_normalize = partial(F.normalize, p=2, dim=-1)
    else:
        truncation = args.truncation
        trunc = generator.mean_latent(4096).detach().clone()
        input_is_latent = True
        n_latent = args.n_latent
        latent_normalize = lambda x: x

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
            img_rec, _ = generator([latent_normalize(z.unsqueeze(0).to(device))], 
                                    input_is_latent=input_is_latent, 
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
        filename = args.latent_path.replace('.pkl', '_verify.png')
        imsave(filename, tensor2image(img))
        exit(0)
    
    if args.part is not None:
        assert args.n_part is not None
        seed = args.seed + args.part
        # args.latent_path = args.latent_path.replace('.pkl', f'_{args.part}.pkl')
        args.latent_path = args.latent_path+f'.part{args.part}'
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
        images_ = []
        for mode in ['train', 'test']:
            dataset = datasets.CIFAR100(args.data_root, download=True, transform=transform, train=mode=='train')
            dataset_ = datasets.CIFAR100(args.data_root, download=True, train=mode=='train')
            for i in tqdm(range(len(dataset))):
                img, lbl = dataset[i]
                img_ = dataset_[i][0]  # uint8 [32, 32, 3]
                images.append(img)
                labels.append(lbl)
                images_.append(np.array(img_))
                counter += 1
        images = torch.stack(images)
        images_ = np.stack(images_)
        data['images'] = images
        data['labels'] = labels
        data['uint8'] = images_
        print(f'[{counter} images loaded]')
        with open(args.data_path, 'wb') as f:
            pickle.dump(data, f)

    vgg_loss = VGGLoss(device)

    toggle_grad(generator, False)
    if encoder is not None:
        toggle_grad(encoder, False)

    latents = []
    index = np.arange(len(data['labels']))
    if args.part is not None:
        part_size = int(np.ceil(len(index) / args.n_part))
        index = index[args.part * part_size:(args.part + 1) * part_size]

    for i in tqdm(range(int(np.ceil(len(index) / batch_size)))):
        inds = index[i * batch_size:(i + 1) * batch_size]
        imgs = images[inds]
        imgs = imgs.to(device)

        with torch.no_grad():
            if encoder is not None:
                z0 = encoder(imgs)
            else:
                if trunc is None:
                    z0 = torch.randn(batch_size, latent_dim, device=device)
                else:
                    z0 = trunc.repeat(batch_size, n_latent, 1)
                z0 = z0[:imgs.shape[0]]
            if args.uniform_noise:
                z0 = F.normalize(z0, p=2, dim=-1)

        z = z0.detach().clone()
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.lr_inv)
        for step in range(args.iters_inv):
            imgs_gen, _ = generator([latent_normalize(z)], 
                                    input_is_latent=input_is_latent, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if encoder is None:
                loss_enc = 0
            else:
                z_hat = encoder(imgs_gen)
                loss_enc = F.mse_loss(z0, z_hat) * 2.0
            loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + loss_enc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        z1 = z.detach().cpu().clone()
        z1.requires_grad = False
        latents.append(z1)
    latents = torch.cat(latents, dim=0)
    with open(args.latent_path, 'wb') as f:
        pickle.dump({'latents': latents}, f)