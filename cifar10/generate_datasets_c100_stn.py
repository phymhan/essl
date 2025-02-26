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

SCALE_LOW = np.sqrt(0.2)
SCALE_HIGH = 1

def stn_crop(theta, imgs, constraint='clamp'):
    # if constraint == 'clamp':
    small = 0.001
    theta_clamp = theta.detach().clone()
    theta_clamp[:,0] = torch.clamp(theta_clamp[:,0], SCALE_LOW, SCALE_HIGH)
    theta_clamp[:,4] = torch.clamp(theta_clamp[:,4], SCALE_LOW, SCALE_HIGH)
    scale_x = theta_clamp[:,0]
    scale_y = theta_clamp[:,4]
    margin_x = F.relu(1 - scale_x)
    margin_y = F.relu(1 - scale_y)
    theta_clamp[:,2] = torch.clamp(theta_clamp[:,2], -margin_x, margin_x)
    theta_clamp[:,5] = torch.clamp(theta_clamp[:,5], -margin_y, margin_y)
    theta_clamp[:,1] = torch.clamp(theta_clamp[:,1], -small, small)
    theta_clamp[:,3] = torch.clamp(theta_clamp[:,1], -small, small)

    theta = theta + (theta_clamp - theta).detach()

    grid = F.affine_grid(theta.view(-1, 2, 3), imgs.size())
    imgs_gen = F.grid_sample(imgs, grid)

    return imgs_gen

class GanStnWrapper(nn.Module):
    def __init__(self, generator, truncation=0.7, trunc=None, stn_type='crop', stn_grad_scale=1):
        super().__init__()
        self.generator = generator
        self.truncation = truncation
        self.trunc = trunc
        self.stn_type = stn_type
    
    def forward(self, w, theta):
        x, _ = self.generator(
            [w],
            input_is_latent=True,
            truncation=self.truncation,
            truncation_latent=self.trunc,
            randomize_noise=False
        )
        if self.stn_type == 'crop':
            x_ = stn_crop(theta, x)
        elif self.stn_type == 'affine':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return x_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=100)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--data_path', type=str, default='data_c100/c100_data.pkl')
    parser.add_argument('--latent_path', type=str, default='data_c100/c100_latent.pkl')
    parser.add_argument('--dest_data_path', type=str, default='data_c100/c100_gen.pkl')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=0.5)
    parser.add_argument('--init_noise_scale', type=float, default=0.005)
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
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--start_from_orig', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--g_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0423_gan_c100/weight/220000.pt')
    parser.add_argument('--model_path', type=str, default='../../SupContrast/logs/0428_c100_bs=512_base/weights/last.pth')
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--stn_type', type=str, default='crop')
    parser.add_argument("--stn_grad_scale", type=float, default=1)
    args = parser.parse_args()

    # utils.fix_seed(args.seed)
    os.makedirs(Path(args.dest_data_path).parent, exist_ok=True)
    if args.merge:
        assert args.n_part is not None
        views = []
        for part in range(args.n_part):
            # dest_data_path = args.dest_data_path.replace('.pkl', f'_{part}.pkl')
            dest_data_path = args.dest_data_path+f'.part{part}'
            with open(dest_data_path, 'rb') as f:
                views.append(pickle.load(f)['views'])
        views = torch.cat(views, dim=1)  # NOTE: batch is second dim
        with open(args.dest_data_path, 'wb') as f:
            pickle.dump({'seed': args.seed, 'n_part': args.n_part, 'n_views': args.n, 'views': views}, f)
        exit(0)

    # name = args.name
    args.clamp = True
    
    device = 'cuda'
    image_size = args.image_size
    tol = args.tol

    # g_model_path = '/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0423_gan_c100/weight/220000.pt'
    g_ckpt = torch.load(args.g_ckpt, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    truncation = args.truncation
    trunc = generator.mean_latent(4096).detach().clone()

    batch_size = args.batch_size

    if args.part is not None:
        assert args.n_part is not None
        seed = args.seed + args.part
        args.dest_data_path = args.dest_data_path+f'.part{args.part}'
    else:
        seed = args.seed
    utils.fix_seed(seed)

    vgg_loss = VGGLoss(device)
    eps1 = args.eps1
    eps2 = args.eps2
    p = args.p
    n = args.n

    # Define SimCLR encoder

    if args.no_proj:
        raise NotImplementedError
    else:
        from main import Branch
        normalize = partial(F.normalize, dim=1)
        # prefix = 'proj'
        from resnet_big import SupConResNet
        model = SupConResNet(name='resnet50')
        state_dict = torch.load(args.model_path, map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

    # ====================== #
    model.eval()  # NOTE: eval!!!
    # ====================== #

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    toggle_grad(model, False)
    toggle_grad(generator, False)

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

    views_image = []
    views_latent = []

    generator_stn = GanStnWrapper(generator, truncation, trunc)

    index = np.arange(len(data['labels']))
    if args.part is not None:
        part_size = int(np.ceil(len(index) / args.n_part))
        index = index[args.part * part_size:(args.part + 1) * part_size]

    for i in tqdm(range(int(np.ceil(len(index) / batch_size)))):
        inds = index[i * batch_size:(i + 1) * batch_size]
        bsz = len(inds)
        # st()
        imgs = images[inds].clone()
        # lbls = [labels[ind] for ind in inds]
        z1 = latents[inds].clone()
        imgs = imgs.to(device)
        z1 = z1.to(device)

        with torch.no_grad():
            imgs_rec, _ = generator([z1],
                                    input_is_latent=True,
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if args.clamp:
                imgs_rec = torch.clamp(imgs_rec, -1, 1)
            if args.start_from_orig:
                h_img = model(encoder_input_transform(imgs))
            else:
                h_img = model(encoder_input_transform(imgs_rec))  # start from the reconstructed images
            h_img = normalize(h_img.squeeze()).detach()
            h_img = h_img.repeat_interleave(n, dim=0)

        z = z1.repeat_interleave(n, dim=0)
        z_noise = args.init_noise_scale * torch.randn_like(z)  # [b*n, L, D]
        if not args.add_noise:
            z_noise[::n, ...] = 0  # make sure the first one is accurate
        z = z + z_noise
        z.requires_grad = True

        # ---------- stn ----------
        theta = torch.zeros(z1.shape[0], 6, device=device).repeat_interleave(n, dim=0)
        theta[:,0] = 1
        theta[:,4] = 1
        theta_noise = args.init_noise_scale * torch.randn_like(theta)  # [b*n, 6]
        theta_noise[::n, ...] = 0  # make sure the first one is accurate
        theta = theta + theta_noise
        theta.requires_grad = True
        # params = itertools.chain(params, [theta])
        params = [
            {'params': [z]},
            {'params': [theta], 'lr': args.stn_grad_scale * args.lr},
        ]
        # ---------- stn ----------

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr)
        elif args.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(params, max_iter=500)  # TODO: max_iter

        done = False
        for step in range(args.iters):
            # imgs_gen, _ = generator([z],
            #                         input_is_latent=True, 
            #                         truncation=truncation,
            #                         truncation_latent=trunc, 
            #                         randomize_noise=False)
            imgs_gen = generator_stn(z, theta)
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
            # imgs_gen, _ = generator([z],
            #                         input_is_latent=True,
            #                         truncation=truncation,
            #                         truncation_latent=trunc,
            #                         randomize_noise=False)  # after the update
            imgs_gen = generator_stn(z, theta)
            if args.clamp:
                imgs_gen = torch.clamp(imgs_gen, -1, 1)
        imgs_gen = imgs_gen.view(bsz, n, 3, image_size, image_size)
        z = rearrange(z, '(b n) m d -> b n m d', b = bsz, n = n)
        views_latent.append(z.detach().cpu().data)
        views_image.append(imgs_gen.detach().cpu().data)

    views_image = torch.cat(views_image, dim=0)
    views_latent = torch.cat(views_latent, dim=0)
    views = {'n_views': n}
    views['views'] = views_image.transpose(0, 1)  # [n_views, B, 3, H, W]
    views['latents'] = views_latent.transpose(0, 1)  # [n_views, B, M, D]
    with open(os.path.join(args.dest_data_path), 'wb') as f:
        pickle.dump(views, f)
