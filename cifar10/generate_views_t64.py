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
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_root', type=str, default='logs_view_t64')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--which_model', type=str, default='simclr1')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=1)
    parser.add_argument('--init_noise_scale', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--lam2', type=float, default=0, help='weight of distance regularization')
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--objective', type=str, default='norm', choices=['norm', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l2', 'l1', 'hinge'])
    parser.add_argument('--method', type=str, default='gd', choices=['gd', 'fgsm', 'vat', 'vg2'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--g_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-pytorch/logs/0401_gan_t64/weight/latest.pt')
    parser.add_argument('--e_ckpt', type=str, default='/research/cbim/medical/lh599/active/stylegan2-encoder-pytorch/checkpoint_t64/encoder_980000.pt')
    parser.add_argument('--n_latent', type=int, default=8)

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    device = 'cuda'
    image_size = args.image_size
    tol = 1e-5

    os.makedirs(args.log_root, exist_ok=True)

    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # use a cached file
    import string
    import pickle
    cache = ''.join(random.choice(string.ascii_uppercase) for i in range(8))
    cache = args.log_root + '/' + cache + '.pkl'
    if os.path.exists(cache):
        print('Loading cached data')
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        imgs = data['imgs']
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join('/research/cbim/medical/lh599/active/BigGAN/data/tiny-imagenet-200', 'train'),
            transform=transform
        )
        loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))
        imgs, _ = next(loader)
        with open(cache, 'wb') as f:
            pickle.dump({'imgs': imgs}, f)
        data = {'imgs': imgs}

    imgs = imgs.to(device)

    name = args.name
    log_root = Path(args.log_root)
    log_dir = log_root / name
    os.makedirs(log_dir, exist_ok=True)

    USE_HTML = True
    log_web_dir = log_dir / 'web'
    webpage = None
    if USE_HTML:
        import utils_html
        webpage = utils_html.initialize_webpage(log_web_dir, 'ViewMaker: ' + args.name, resume=False)

    g_ckpt = torch.load(args.g_ckpt, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    if args.e_ckpt:
        e_ckpt = torch.load(args.e_ckpt, map_location=device)

        encoder = Encoder(image_size, latent_dim).to(device)
        encoder.load_state_dict(e_ckpt['e'])
        encoder.eval()
        print('[encoder loaded]')
    else:
        encoder = None
        print('[no encoder loaded]')

    truncation = args.truncation
    trunc = generator.mean_latent(4096).detach().clone()

    imgs_real = torch.cat([img for img in imgs], dim=1)

    if data is None or 'z' not in data:
        with torch.no_grad():
            if encoder is None:
                z0 = trunc.repeat(batch_size, args.n_latent, 1)
            else:
                z0 = encoder(imgs)
            imgs_gen, _ =  generator([z0], 
                                    input_is_latent=True,
                                    truncation=truncation,
                                    truncation_latent=trunc,
                                    randomize_noise=False)

        imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
        imsave(log_dir / f'rec_step{0}.png', tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)))

        # In-domain inversion
        vgg_loss = VGGLoss(device)
        z = z0.detach().clone()

        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.lr_inv)

        for step in range(args.iters_inv):
            imgs_gen, _ = generator([z], 
                                    input_is_latent=True, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if encoder is None:
                loss_enc = 0
            else:
                z_hat = encoder(imgs_gen)
                loss_enc = F.mse_loss(z0, z_hat)*2.0
            loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + loss_enc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f'[inv] step {step}/{args.iters_inv} loss {loss.item():.4f}')
                imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
                imsave(log_dir / f'rec_step{step}.png', tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)))

        imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
        imsave(log_dir / f'rec_{step+1}.png', tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)))

        data['z'] = z.data.cpu()

        with open(cache, 'wb') as f:
            pickle.dump(data, f)
    
    else:
        z = data['z']
        z = z.to(device)
        print('[loaded z]')
        # imgs_gen, _ = generator([z], 
        #                         input_is_latent=True, 
        #                         truncation=truncation,
        #                         truncation_latent=trunc, 
        #                         randomize_noise=False)

    z1 = z.detach().clone()

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    
    # Define SimCLR encoder
    normalize = partial(F.normalize, dim=1)

    # load pretrained simclr encoder
    if args.which_model == 'simclr1':
        # tiny1, supcon codebase
        from resnet_big import SupConResNet
        model = SupConResNet(name='resnet50')
        state_dict = torch.load('../../SupContrast/logs/0430_t64_bs=512_base_2/weights/ckpt_epoch_1000.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    elif args.which_model == 'simclr2':
        # tiny2, self-supervised codebase
        from selfsup.contrastive import Contrastive
        cfg = Config(
            emb=128,
            tau=0.5,
            norm=True,
            arch='resnet18',
            dataset='tiny_in',
            knn=5,
            num_samples=2,
            eval_head=False,
            head_layers=2,
            head_size=1024,
            add_bn=True,
        )
        model = Contrastive(cfg)
        model.cuda()
        state_dict = torch.load('../../self-supervised/logs/tiny-in_base/weights/contrastive_tiny_in_999.pt', map_location='cpu')['model']
        model.load_state_dict(state_dict, strict=True)

    if args.eval:
        print('eval mode')
        model.eval()

    eps1 = args.eps1
    eps2 = args.eps2
    p = args.p
    n = args.n

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
    if args.clamp:
        imgs_rec = torch.clamp(imgs_rec, -1, 1)
    imgs_recon = torch.cat([img_rec for img_rec in imgs_rec], dim=1)
    imgs_blank = torch.ones_like(imgs_recon)[:,:,:8]

    z.requires_grad = True
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([z], lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([z], lr=args.lr)
    elif args.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS([z], max_iter=500)  # TODO: max_iter

    toggle_grad(model, False)
    toggle_grad(generator, False)

    # h_img = model(imgs_rep)  # precompute
    h_img = model(encoder_input_transform(imgs_rec).repeat_interleave(n, dim=0))  # start from the reconstructed images
    h_img = normalize(h_img.squeeze()).detach()

    losses = []
    print('generating views...')
    done = False
    for step in range(args.iters):
        imgs_gen, _ = generator([z],
                                input_is_latent=True, 
                                truncation=truncation,
                                truncation_latent=trunc, 
                                randomize_noise=False)
        if args.clamp:
            imgs_gen = torch.clamp(imgs_gen, -1, 1)
        h_gen = model(encoder_input_transform(imgs_gen))
        h_gen = normalize(h_gen.squeeze())

        pdist = torch.cdist(h_gen.view(batch_size, n, -1), h_gen.view(batch_size, n, -1), p=p)
        pdist = pdist * n / (n-1)
        loss_reg = torch.mean(F.relu(eps2 - torch.mean(pdist.view(batch_size*n, n), dim=1)))

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
        elif args.method == 'vg2':
            z = z + args.eps1 * torch.randn_like(z)
            done = True
        
        if args.objective == 'norm':
            losses.append(torch.norm(h_gen - h_img, dim=1, p=p).mean().item())
        elif args.objective == 'cosine':
            losses.append(torch.sum(h_gen * h_img, dim=1).mean().item())

        with torch.no_grad():
            if done or step == 0 or step == args.iters - 1 or (step + 1) % args.save_every == 0:
                imgs_gen, _ = generator([z],
                                        input_is_latent=True,
                                        truncation=truncation,
                                        truncation_latent=trunc,
                                        randomize_noise=False)  # after the update
                if args.clamp:
                    imgs_gen = torch.clamp(imgs_gen, -1, 1)
                h_gen = model(encoder_input_transform(imgs_gen))
                h_gen = normalize(h_gen.squeeze())

                if args.objective == 'norm':
                    print(f'step: {step+1}, loss: {loss.item()}, norm: {torch.norm(h_gen - h_img, dim=1, p=p).mean().item()}, pdist: {pdist.mean().item()}')
                elif args.objective == 'cosine':
                    print(f'step: {step+1}, loss: {loss.item()}, cos: {torch.sum(h_gen * h_img, dim=1).mean().item()}, pdist: {pdist.mean().item()}')
                # st()
                # imsave(log_dir / 'debug_clamp.png', tensor2image(torch.cat([xx for xx in torch.clamp(imgs_gen[32:40,...], -1, 1)], dim=2)))

                imgs_gen = imgs_gen.view(batch_size, n, 3, image_size, image_size)
                imgs_fakes = []
                imgs_diffs = []
                for j in range(n):
                    imgs_fakes.append(torch.cat([img_gen for img_gen in imgs_gen[:,j,...]], dim=1))
                    img_diff = torch.cat([img_gen for img_gen in imgs_gen[:,j,...] - imgs_rec], dim=1)
                    imgs_diffs.append((img_diff - img_diff.min()) / (img_diff.max() - img_diff.min()) * 2 - 1)
                imsave(log_dir / f'view_{step+1:04d}.png', tensor2image(torch.cat([imgs_real, imgs_recon, imgs_blank] + imgs_fakes, dim=2)))
                imsave(log_dir / f'diff_{step+1:04d}.png', tensor2image(torch.cat([imgs_real, imgs_recon, imgs_blank] + imgs_diffs, dim=2)))
                image_tensor = torch.cat([imgs_real, imgs_recon, imgs_blank] + imgs_fakes, dim=2)
                if USE_HTML:
                    if args.objective == 'norm':
                        header = f'step: {step+1}, loss: {loss.item()}, norm: {torch.norm(h_gen - h_img, dim=1, p=p).mean().item()}, pdist: {pdist.mean().item()}'
                    elif args.objective == 'cosine':
                        header = f'step: {step+1}, loss: {loss.item()}, cos: {torch.sum(h_gen * h_img, dim=1).mean().item()}, pdist: {pdist.mean().item()}'
                    webpage.add_header(header)
                    utils_html.save_grid(
                        webpage=webpage,
                        tensor=[(image_tensor + 1) / 2],
                        caption=[f'real recon | views x {n}'],
                        name=f'{step+1:04d}',
                        nrow=[1],
                        width=768,
                    )
        
        if done:
            print(f"loss is {loss.item()}!")
            if args.objective == 'norm':
                print(torch.norm(h_gen - h_img, dim=1, p=p).mean().item(), pdist.mean().item())
            elif args.objective == 'cosine':
                print(torch.sum(h_gen * h_img, dim=1).mean().item(), pdist.mean().item())
            break

    losses = np.array(losses)

    plt.plot(losses)
    plt.xlabel('steps')
    if args.objective == 'norm':
        plt.ylabel(f'L{p}')
    elif args.objective == 'cosine':
        plt.ylabel(f'cos')
    plt.savefig(log_dir / f'loss_plot.png')
