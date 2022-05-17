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
from einops import rearrange
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
    parser.add_argument('--view_path', type=str, default='data_c100/c100_latent_view.pkl')
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--fake', action='store_true')

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    device = 'cuda'
    image_size = args.image_size
    tol = 1e-3

    batch_size = args.batch_size

    if not args.name.endswith('.png'):
        args.name += '.png'

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

    f_norms = []
    g_norms = []

    z_norms = []
    w_norms = []

    f_sims = []
    g_sims = []

    z_sims = []
    w_sims = []

    view_latents = []

    for i in track(range(int(args.N / batch_size))):
        inds = index[i * batch_size: (i + 1) * batch_size]

        bsz = len(inds)

        if args.fake:
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
        else:
            imgs = images[inds].clone()
            # lbls = [labels[ind] for ind in inds]
            z1 = latents[inds].clone()
            imgs = imgs.to(device)
            z1 = z1.to(device)
            z1 = latent_normalize(z1)

        w0 = z1.clone().cpu()

        with torch.no_grad():
            imgs_rec, _ = generator([z1],
                                    input_is_latent=input_is_latent,
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if args.clamp:
                imgs_rec = torch.clamp(imgs_rec, -1, 1)
            if args.start_from_recon:
                h_img = model(encoder_input_transform(imgs_rec))  # start from the reconstructed images
            else:
                h_img = model(encoder_input_transform(imgs))
            h_img = normalize(h_img.squeeze()).detach()
            h_img = h_img.repeat_interleave(n, dim=0)

        z = z1.repeat_interleave(n, dim=0)
        z_noise = args.init_noise_scale * torch.randn_like(z)  # [b*n, L, D]
        z_noise[::n, ...] = 0  # make sure the first one is accurate
        z = z + z_noise

        z.requires_grad = True
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam([z], lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([z], lr=args.lr)
        elif args.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS([z], max_iter=500)  # TODO: max_iter

        done = False
        for step in range(args.iters):
            imgs_gen, _ = generator([latent_normalize(z)],
                                    input_is_latent=input_is_latent, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if args.clamp:
                imgs_gen = torch.clamp(imgs_gen, -1, 1)
            h_gen = model(encoder_input_transform(imgs_gen))
            h_gen = normalize(h_gen.squeeze())
            if args.lam2 > 0:
                pdist = torch.cdist(h_gen.view(bsz, n, -1), h_gen.view(bsz, n, -1), p=p)
                pdist = pdist * n / (n-1)
                loss_reg = torch.mean(F.relu(eps2 - torch.mean(pdist.view(bsz*n, n), dim=1)))
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
        with torch.no_grad():
            imgs_gen, _ = generator([latent_normalize(z)],
                                    input_is_latent=input_is_latent,
                                    truncation=truncation,
                                    truncation_latent=trunc,
                                    randomize_noise=False)  # after the update
            if args.clamp:
                imgs_gen = torch.clamp(imgs_gen, -1, 1)
            
            h_gen = model(encoder_input_transform(imgs_gen))
            h_gen = F.normalize(h_gen, dim=-1)

            w1 = z.clone().cpu()
        
        df = (h_gen - h_img).detach().cpu()
        dg = (w1 - w0.repeat_interleave(n, dim=0)).detach().cpu().reshape(bsz * n, -1)

        df_norm = torch.norm(df, p=2, dim=-1)
        f_norms.append(df_norm)
        f_sims.append(F.cosine_similarity(h_gen, h_img, dim=-1).detach().cpu())

        dg_norm = torch.norm(dg, p=2, dim=-1)
        g_norms.append(dg_norm)
        g_sims.append(F.cosine_similarity(w1.reshape(bsz * n, -1), w0.repeat_interleave(n, dim=0).reshape(bsz * n, -1), dim=-1).detach().cpu())

        with torch.no_grad():
            w_ = w0.repeat_interleave(n, dim=0).cuda()
            w_noise = torch.randn_like(w_) * args.init_noise_scale
            w_n = w_ + w_noise
            w_n = latent_normalize(w_n)
            x_n, _ = generator([w_n],input_is_latent=input_is_latent,truncation=truncation,truncation_latent=trunc,randomize_noise=False)
            z_n = model(encoder_input_transform(x_n))
            z_n = F.normalize(z_n, p=2, dim=-1)
            z_orig = h_img
        
        dz = (z_n - z_orig).cpu()
        dz_norm = torch.norm(dz, p=2, dim=-1)
        z_norms.append(dz_norm)
        z_sims.append(F.cosine_similarity(z_n, z_orig, dim=-1).cpu())

        dw = w_noise.reshape(batch_size * n, -1).cpu()
        dw_norm = torch.norm(dw, p=2, dim=-1)
        w_norms.append(dw_norm)
        w_sims.append(F.cosine_similarity(w_.reshape(batch_size * n, -1), w_n.reshape(batch_size * n, -1), dim=-1).cpu())

        if n_latent == 1:
            z = rearrange(z, '(b n) d -> b n d', b = bsz, n = n)
        else:
            z = rearrange(z, '(b n) m d -> b n m d', b = bsz, n = n)
        view_latents.append(z.detach().cpu().data)

    view_latents = torch.cat(view_latents, dim=0)
    f_norms = torch.cat(f_norms, dim=0).numpy()
    g_norms = torch.cat(g_norms, dim=0).numpy()
    z_norms = torch.cat(z_norms, dim=0)
    z_norms = z_norms.numpy()
    w_norms = torch.cat(w_norms, dim=0)
    w_norms = w_norms.numpy()

    f_sims = torch.cat(f_sims, dim=0).numpy()
    g_sims = torch.cat(g_sims, dim=0).numpy()
    z_sims = torch.cat(z_sims, dim=0).numpy()
    w_sims = torch.cat(w_sims, dim=0).numpy()

    if args.loss_type != 'hinge' and args.N >= 1000:
        views = {'n_views': n}
        views['latents'] = view_latents.transpose(0, 1)  # [n_views, B, M, D]
        pickle_file_name = args.name.replace('.png', '_view_latents.pkl')
        with open(os.path.join(pickle_file_name), 'wb') as f:
            pickle.dump(views, f)
    
    pickle_file_name = args.name.replace('.png', f'_stats.pkl')
    with open(os.path.join(pickle_file_name), 'wb') as f:
        pickle.dump({
            'f_norms': f_norms, 'f_sims': f_sims, 'g_norms': g_norms, 'g_sims': g_sims,
            'z_norms': z_norms, 'z_sims': z_sims, 'w_norms': w_norms, 'w_sims': w_sims,
            }, f)

    bins = 20
    # ---------- overlay norm ---------- #
    plt.figure()
    plt.subplot(1,2,1)
    # plt.hist(g_norms, bins=bins, density=True)
    # plt.hist(w_norms, bins=bins, density=True)
    plt.hist([g_norms, w_norms], bins=bins, density=True)
    plt.legend(['VG1', 'VG2'])
    plt.title('GAN')
    
    plt.subplot(1,2,2)
    # plt.hist(f_norms, bins=bins, density=True)
    # plt.hist(z_norms, bins=bins, density=True)
    plt.hist([f_norms, z_norms], bins=bins, density=True)
    plt.legend(['VG1', 'VG2'])
    plt.title('SimCLR')

    plt.savefig(args.name.replace('.png', '_norms_overlay.png'))
    
    # ---------- separate norm ---------- #
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(g_norms, bins=bins, density=True)
    plt.title('GAN')
    plt.subplot(1,2,2)
    plt.hist(f_norms, bins=bins, density=True)
    plt.title('SimCLR')
    plt.savefig(args.name.replace('.png', '_norms_VG1.png'))

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(w_norms, bins=bins, density=True)
    plt.title('GAN')
    plt.subplot(1,2,2)
    plt.hist(z_norms, bins=bins, density=True)
    plt.title('SimCLR')
    plt.savefig(args.name.replace('.png', '_norms_VG2.png'))

    # ---------- overlay cosine ---------- #
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist([g_sims, w_sims], bins=bins, density=True)
    plt.legend(['VG1', 'VG2'])
    plt.title('GAN')
    
    plt.subplot(1,2,2)
    plt.hist([f_sims, z_sims], bins=bins, density=True)
    plt.legend(['VG1', 'VG2'])
    plt.title('SimCLR')

    plt.savefig(args.name.replace('.png', '_sims_overlay.png'))
