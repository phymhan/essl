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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=100)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--log_root', type=str, default='logs_gen')
    # parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--model_dir', type=str, default='checkpoint')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--data_dest_root', type=str, default='data/cifar10_gen')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=1)
    parser.add_argument('--init_noise_scale', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--lam2', type=float, default=0, help='weight of distance regularization')
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--objective', type=str, default='norm', choices=['norm', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l2', 'l1', 'hinge'])
    parser.add_argument('--method', type=str, default='gd', choices=['gd', 'fgsm', 'vat'])

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    # name = args.name
    
    device = 'cuda'
    image_size = args.image_size
    tol = 1e-3

    g_model_path = '../pretrained/stylegan2-c10_g.pt'
    g_ckpt = torch.load(g_model_path, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    e_model_path = '../pretrained/stylegan2-c10_e.pt'
    e_ckpt = torch.load(e_model_path, map_location=device)

    encoder = Encoder(image_size, latent_dim).to(device)
    encoder.load_state_dict(e_ckpt['e'])
    encoder.eval()
    print('[encoder loaded]')

    truncation = args.truncation
    trunc = generator.mean_latent(4096).detach().clone()

    # create data folders
    os.makedirs(args.data_dest_root, exist_ok=True)
    os.makedirs(os.path.join(args.data_dest_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dest_root, 'test'), exist_ok=True)

    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    vgg_loss = VGGLoss(device)
    eps1 = args.eps1
    eps2 = args.eps2
    p = args.p
    n = args.n

    # Define SimCLR encoder
    if args.no_proj:
        if args.objective == 'norm':
            normalize = lambda x: x
        elif args.objective == 'cosine':
            normalize = partial(F.normalize, dim=1)
        prefix = 'noproj'
        from resnet import resnet18
        model = resnet18(pretrained=False, num_classes=10)
        checkpoint = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=True)
        # assert log.missing_keys == ['fc.weight', 'fc.bias']
        # model = ResNetWrapper(model).to(device)
        model.to(device)
    else:
        from main import Branch
        normalize = partial(F.normalize, dim=1)
        prefix = 'proj'
        args_simclr = Config(dim_proj='2048,2048', dim_pred=512, loss='simclr')
        model = Branch(args_simclr).to(device)
        saved_dict = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')['state_dict']
        model.load_state_dict(saved_dict, strict=True)

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    toggle_grad(model, False)
    toggle_grad(generator, False)
    toggle_grad(encoder, False)

    for mode in ['train', 'test']:
        counter = 0
        for view in range(args.n + 1):
            for y in range(10):
                os.makedirs(os.path.join(args.data_dest_root, mode, f'view_{view}', f'{y}'), exist_ok=True)
        dataset = datasets.CIFAR10(root=args.data_root, download=True, transform=transform, train=mode=='train')
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        # NOTE: I put shuffle=False in case batch norm needs random samples
        for batch in tqdm(loader):
            imgs, labels = batch
            imgs = imgs.to(device)

            with torch.no_grad():
                z0 = encoder(imgs)
                imgs_gen, _ =  generator([z0], 
                                        input_is_latent=True,
                                        truncation=truncation,
                                        truncation_latent=trunc,
                                        randomize_noise=False)

            # In-domain inversion
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

            z1 = z.detach().clone()
            z = z1.detach().clone()
            z = z.repeat_interleave(n, dim=0)
            z_noise = args.init_noise_scale * torch.randn_like(z)  # [b*n, L, D]
            z_noise[::n, ...] = 0  # make sure the first one is accurate
            z = z + z_noise

            imgs_rep = imgs.repeat_interleave(n, dim=0)
            imgs_rec, _ = generator([z1],
                                    input_is_latent=True,
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            imgs_recon = torch.cat([img_rec for img_rec in imgs_rec], dim=1)
            imgs_blank = torch.ones_like(imgs_recon)[:,:,:8]

            z.requires_grad = True
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam([z], lr=args.lr)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([z], lr=args.lr)
            elif args.optimizer == 'lbfgs':
                optimizer = torch.optim.LBFGS([z], max_iter=500)  # TODO: max_iter

            # h_img = model(imgs_rep)  # precompute
            h_img = model(encoder_input_transform(imgs_rec).repeat_interleave(n, dim=0))  # start from the reconstructed images
            h_img = normalize(h_img.squeeze()).detach()

            done = False
            for step in range(args.iters):
                imgs_gen, _ = generator([z],
                                        input_is_latent=True, 
                                        truncation=truncation,
                                        truncation_latent=trunc, 
                                        randomize_noise=False)

                h_gen = model(encoder_input_transform(imgs_gen))
                h_gen = normalize(h_gen.squeeze())

                if args.lam2 > 0:
                    pdist = torch.cdist(h_gen.view(batch_size, n, -1), h_gen.view(batch_size, n, -1), p=p)
                    pdist = pdist * n / (n-1)
                    loss_reg = torch.mean(F.relu(eps2 - torch.mean(pdist.view(batch_size*n, n), dim=1)))
                else:
                    loss_reg = 0

                if args.loss_type == 'l2':
                    if args.objective == 'norm':
                        diff = torch.norm(h_gen - h_img, dim=1, p=p) - eps1
                    elif args.objective == 'cosine':
                        diff = torch.sum(h_gen * h_img, dim=1) - eps1
                    loss = torch.mean(diff ** 2) + args.lam2 * loss_reg
                elif args.loss_type == 'hinge':
                    if args.objective == 'norm':
                        diff = F.relu(eps1 - torch.norm(h_gen - h_img, dim=1, p=p))
                    elif args.objective == 'cosine':
                        diff = F.relu(torch.sum(h_gen * h_img, dim=1) - eps1)
                    loss = torch.mean(diff) + args.lam2 * loss_reg

                if args.method == 'gd':
                    if loss.item() > tol:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        done = True
                elif args.method == 'fgsm':
                    optimizer.zero_grad()
                    loss.backward()
                    z_delta = z.grad.data.sign()
                    z = z - args.lr * z_delta
                    done = True

                if done:
                    break
            
            # get samples
            imgs_gen, _ = generator([z],
                                    input_is_latent=True,
                                    truncation=truncation,
                                    truncation_latent=trunc,
                                    randomize_noise=False)  # after the update
            imgs_gen = imgs_gen.view(batch_size, n, 3, image_size, image_size)

            # save images
            for j in range(batch_size):
                counter += 1
                y = f"{labels[j].item()}"
                imsave(os.path.join(args.data_dest_root, mode, f'view_{0}', f'{y}', f'{counter:05d}.png'),
                    tensor2image(imgs[j]))
                for kk in range(n):
                    imsave(os.path.join(args.data_dest_root, mode, f'view_{kk+1}', f'{y}', f'{counter:05d}.png'),
                        tensor2image(imgs_gen[j,kk]))
        print(f'{mode}: {counter} images')