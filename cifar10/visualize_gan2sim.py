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

from utils import toggle_grad, image2tensor, tensor2image, imshow, imsave, Config
import pickle

from rich.progress import track
from tqdm import tqdm

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

def get_transform_list(which_transform, size=32):
    if isinstance(which_transform, str):
        which_transform = which_transform.replace(',', '+').split('+')
    transform_list = []
    for t in which_transform:
        t = t.lower()
        if t == 'resizedcrop':
            transform_list.append(T.RandomResizedCrop(size=size, scale=(0.2, 1.0)))
        elif t == 'resizedcrophalf':
            transform_list.append(T.RandomResizedCrop(size=size//2, scale=(0.2, 1.0)))
        elif t == 'horizontalflip':
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        elif t == 'colorjitter':
            transform_list.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        elif t == 'grayscale':
            transform_list.append(T.RandomGrayscale(p=0.2))
    return transform_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_root', type=str, default='logs_view_c100')
    parser.add_argument('--name', type=str, default='visualize')
    parser.add_argument('--model_dir', type=str, default='checkpoint')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=1)
    parser.add_argument('--init_noise_scale', type=float, default=0.2)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--lam2', type=float, default=0, help='weight of distance regularization')
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--objective', type=str, default='norm', choices=['norm', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l2', 'l1', 'hinge'])
    parser.add_argument('--method', type=str, default='gd', choices=['gd', 'fgsm', 'vat'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--g_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0423_gan_c100/weight/220000.pt')
    parser.add_argument('--e_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-encoder-pytorch/checkpoint_c100/encoder_980000.pt')
    parser.add_argument('--n_latent', type=int, default=8)
    parser.add_argument('--invert', action='store_true')
    parser.add_argument("--augment_leaking", action='store_true', help="apply non-differentiable, 'leaking' augmentation")
    parser.add_argument("--which_transform", type=str, default='resizedcrop+horizontalflip+colorjitter+grayscale')
    parser.add_argument("--start_from_recon", action='store_true')
    parser.add_argument("--cache_prefix", type=str, default='')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--uniform_noise', action='store_true')
    parser.add_argument('--data_path', type=str, default='data_c100/c100_data.pkl')
    parser.add_argument('--latent_path', type=str, default='data_c100/c100_latent.pkl')
    parser.add_argument('--view_path', type=str, default='data_c100/c100_gen.pkl')
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--fake', action='store_true')

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    device = 'cuda'
    image_size = args.image_size
    tol = 1e-5

    batch_size = args.batch_size

    # load data
    assert os.path.exists(args.data_path)
    assert os.path.exists(args.latent_path)
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    images = data['images']
    labels = data['labels']
    print('[data loaded]')
    with open(args.latent_path, 'rb') as f:
        latents = pickle.load(f)['latents']
    print('[latents loaded]')

    index = np.random.choice(np.arange(0, len(images)), size=args.N, replace=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # input transform
    if args.dataset == 'cifar10':
        encoder_input_transform = T.Compose(
            [
                T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )
    elif args.dataset == 'cifar100':
        encoder_input_transform = T.Compose(
            [
                T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    else:
        raise NotImplementedError
    
    # Define SimCLR encoder
    if args.dataset == 'cifar10':
        from main import Branch
        normalize = partial(F.normalize, dim=1)
        prefix = 'proj'
        args_simclr = Config(dim_proj='2048,2048', dim_pred=512, loss='simclr')
        model = Branch(args_simclr).to(device)
        saved_dict = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')['state_dict']
        model.load_state_dict(saved_dict, strict=True)
    elif args.dataset == 'cifar100':
        normalize = partial(F.normalize, dim=1)
        from resnet_big import SupConResNet
        model = SupConResNet(name='resnet50')
        state_dict = torch.load('../../SupContrast/logs/0428_c100_bs=512_base/weights/last.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    else:
        raise NotImplementedError

    # if args.eval:
    print('eval mode')
    model.eval()

    g_ckpt = torch.load(args.g_ckpt, map_location=device)
    latent_dim = g_ckpt['args'].latent
    assert latent_dim == args.latent_dim
    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

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
        n_latent = 1 if args.fake else args.n_latent
        latent_normalize = lambda x: x

    eps1 = args.eps1
    eps2 = args.eps2
    p = args.p
    n = args.n

    ############################################################################

    z_norms = []
    w_norms = []

    with torch.no_grad():
        for i in track(range(int(args.N / batch_size))):
            inds = index[i * batch_size: (i + 1) * batch_size]

            if args.fake:
                bsz = len(inds)
                with torch.no_grad():
                    if args.uniform_noise:
                        noise = torch.randn(bsz, latent_dim).to(device)
                        z1 = latent_normalize(noise)
                        imgs, _ = generator([z1],
                                            input_is_latent=False,
                                            truncation=truncation,
                                            truncation_latent=trunc, 
                                            randomize_noise=False)
                    else:
                        noise = torch.randn(bsz, latent_dim).to(device)
                        z1 = generator.get_latent(noise)
                        imgs, _ = generator([z1],
                                            input_is_latent=True,
                                            truncation=truncation,
                                            truncation_latent=trunc, 
                                            randomize_noise=False)
                    imgs = torch.clamp(imgs, -1, 1)
                x0 = imgs.detach().clone()
                w = z1.detach().clone()
            else:
                w = latent_normalize(latents[inds].cuda())
                x0 = images[inds].cuda()

            w_ = w.repeat_interleave(n, dim=0)
            w_noise = torch.randn_like(w_) * args.init_noise_scale
            w_n = w_ + w_noise
            w_n = latent_normalize(w_n)
            x_n, _ = generator([w_n],input_is_latent=input_is_latent,truncation=truncation,truncation_latent=trunc,randomize_noise=False)
            x_rec, _ = generator([latent_normalize(w_)],input_is_latent=input_is_latent,truncation=truncation,truncation_latent=trunc,randomize_noise=False)
            x_rec = torch.clamp(x_rec, min=-1, max=1)

            if args.start_from_recon:
                z_orig = model(encoder_input_transform(x_rec))
            else:
                z_orig = model(encoder_input_transform(x0.repeat_interleave(n, dim=0)))
            z_orig = F.normalize(z_orig, p=2, dim=-1)

            z_n = model(encoder_input_transform(x_n))
            z_n = F.normalize(z_n, p=2, dim=-1)

            dz = (z_n - z_orig).cpu()
            dz_norm = torch.norm(dz, p=2, dim=-1)
            z_norms.append(dz_norm)

            dw = w_noise.reshape(batch_size * n, -1).cpu()
            dw_norm = torch.norm(dw, p=2, dim=-1)
            w_norms.append(dw_norm)
    
    z_norms = torch.cat(z_norms, dim=0)
    z_norms = z_norms.numpy()
    w_norms = torch.cat(w_norms, dim=0)
    w_norms = w_norms.numpy()
    
    # plt.hist(z_norms, bins=10)
    # plt.savefig(args.name)

    plt.subplot(1,2,1)
    plt.hist(w_norms, bins=16)
    plt.title('w_norms')
    plt.subplot(1,2,2)
    plt.hist(z_norms, bins=16)
    plt.title('z_norms')
    plt.savefig(args.name)
