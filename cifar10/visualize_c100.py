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
    parser.add_argument('--init_noise_scale', type=float, default=0.001)
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

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    device = 'cuda'
    image_size = args.image_size
    tol = 1e-5

    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # use a cached file
    import string
    import pickle
    cache = ''.join(random.choice(string.ascii_uppercase) for i in range(8))
    cache = args.cache_prefix + cache
    cache = args.log_root + '/' + cache + '.pkl'
    if os.path.exists(cache):
        print('Loading cached data')
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        imgs = data['imgs']
    else:
        dataset = datasets.CIFAR100(root=args.data_root, download=True, transform=transform)
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

    from utils import print_args
    args.log_dir = log_dir
    print_args(parser, args)

    USE_HTML = True
    log_web_dir = log_dir / 'web'
    webpage = None
    if USE_HTML:
        import utils_html
        webpage = utils_html.initialize_webpage(log_web_dir, 'ViewMaker: ' + args.name, resume=False)

    # g_model_path = '../pretrained/stylegan2-c10_g.pt'
    g_ckpt = torch.load(args.g_ckpt, map_location=device)

    latent_dim = g_ckpt['args'].latent
    assert latent_dim == args.latent_dim

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    if args.e_ckpt and os.path.exists(args.e_ckpt):
        # e_model_path = '../pretrained/stylegan2-c10_e.pt'
        e_ckpt = torch.load(args.e_ckpt, map_location=device)

        encoder = Encoder(image_size, latent_dim).to(device)
        encoder.load_state_dict(e_ckpt['e'])
        encoder.eval()
        print('[encoder loaded]')
    else:
        encoder = None
        print('[no encoder loaded]')

    # truncation = args.truncation
    # trunc = generator.mean_latent(4096).detach().clone()
    if args.uniform_noise:
        truncation = 1
        trunc = None
        input_is_latent = False
        n_latent = 1
        latent_normalize = partial(F.normalize, p=2, dim=-1)
        # latent_normalize = lambda x: x
    else:
        truncation = args.truncation
        trunc = generator.mean_latent(4096).detach().clone()
        input_is_latent = True
        n_latent = args.n_latent
        latent_normalize = lambda x: x

    imgs_real = torch.cat([img for img in imgs], dim=1)

    if data is None or 'z' not in data:
        with torch.no_grad():
            if encoder is None:
                if trunc is None:
                    z0 = torch.randn(batch_size, latent_dim, device=device)
                else:
                    z0 = trunc.repeat(batch_size, n_latent, 1)
            else:
                z0 = encoder(imgs)
            if args.uniform_noise:
                z0 = F.normalize(z0, p=2, dim=-1)
            imgs_gen, _ =  generator([z0],
                                    input_is_latent=input_is_latent,
                                    truncation=truncation,
                                    truncation_latent=trunc,
                                    randomize_noise=False)

        imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
        imsave(log_dir / f'rec_step{0}.png', tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)))

        # In-domain inversion
        vgg_loss = VGGLoss(device)
        z = z0.detach().clone()
        z = latent_normalize(z)

        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.lr_inv)

        for step in range(args.iters_inv):
            optimizer.zero_grad()
            imgs_gen, _ = generator([latent_normalize(z)],
                                    input_is_latent=input_is_latent, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            if encoder is None:
                loss_enc = 0
            else:
                z_hat = encoder(imgs_gen)
                loss_enc = F.mse_loss(z0, z_hat)*2.0
            loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + loss_enc
            loss.backward()
            optimizer.step()

            # if args.uniform_noise:
            #     z = F.normalize(z, p=2, dim=-1)

            if step % 100 == 0:
                print(f'[inv] step {step}/{args.iters_inv} loss {loss.item()}')
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
    z1 = latent_normalize(z1)

    w0 = z1.detach().clone()

    if args.augment_leaking:
        vgg_loss = VGGLoss(device)
        transforms_leaking = get_transform_list(args.which_transform, args.image_size)
        transforms_leaking = [transforms.Normalize([-1, -1, -1], [2, 2, 2])] + \
            transforms_leaking + [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transforms_leaking = transforms.Compose(transforms_leaking)
        imgs_leak = torch.stack([transforms_leaking(x) for x in imgs])
        imgs_leaks = torch.cat([img for img in imgs_leak], dim=1)

        z = z1.detach().clone()
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.lr_inv)
        for step in range(args.iters_inv):
            imgs_gen, _ = generator([latent_normalize(z)], 
                                    input_is_latent=input_is_latent, 
                                    truncation=truncation,
                                    truncation_latent=trunc, 
                                    randomize_noise=False)
            loss = F.mse_loss(imgs_gen, imgs_leak) + vgg_loss(imgs_gen, imgs_leak)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
                imsave(log_dir / f'leak_step{step}.png', tensor2image(torch.cat([imgs_leaks, imgs_fakes], dim=2)))
        imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
        imsave(log_dir / f'leak_{step+1}.png', tensor2image(torch.cat([imgs_leaks, imgs_fakes], dim=2)))
        exit(0)

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    
    # Define SimCLR encoder
    if args.no_proj:
        raise NotImplementedError
    else:
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
                            input_is_latent=input_is_latent,
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
    if args.start_from_recon:
        h_img = model(encoder_input_transform(imgs_rec).repeat_interleave(n, dim=0))  # start from the reconstructed images
    else:
        h_img = model(imgs_rep)
    h_img = normalize(h_img.squeeze()).detach()

    losses = []
    print('generating views...')
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
        
        if args.objective == 'norm':
            losses.append(torch.norm(h_gen - h_img, dim=1, p=p).mean().item())
        elif args.objective == 'cosine':
            losses.append(torch.sum(h_gen * h_img, dim=1).mean().item())

        with torch.no_grad():
            if done or step == 0 or step == args.iters - 1 or (step + 1) % args.save_every == 0:
                imgs_gen, _ = generator([latent_normalize(z)],
                                        input_is_latent=input_is_latent,
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

    ############################################################################
    w1 = latent_normalize(z).detach().detach().clone()
    z1 = h_gen
    z0 = h_img
    with open('visualize_c100.pkl', 'wb') as f:
        pickle.dump({'w1': w1, 'w0': w0, 'z1': z1}, f)
    print('visualize_c100.pkl saved!')
    dg = w1 - w0.repeat_interleave(n, dim=0)
    dg = dg.view(batch_size * n, -1).cpu()
    df = z1 - z0
    df = df.view(batch_size * n, -1).cpu()
    dg_norm = torch.norm(dg, dim=1).cpu().numpy()
    df_norm = torch.norm(df, dim=1).cpu().numpy()

    w_noise = torch.randn_like(w0.repeat_interleave(n, dim=0)) * 0.2
    w2 = w0.repeat_interleave(n, dim=0) + w_noise
    x2, _ = generator([latent_normalize(w2)],input_is_latent=input_is_latent,truncation=truncation,truncation_latent=trunc,randomize_noise=False)
    z2 = model(encoder_input_transform(x2))
    z2 = F.normalize(z2, dim=-1)
    df2 = z2 - z0
    df2 = df2.view(batch_size * n, -1).cpu()
    df2_norm = torch.norm(df2, dim=1).cpu().numpy()
    st()
    ############################################################################

    plt.plot(losses)
    plt.xlabel('steps')
    if args.objective == 'norm':
        plt.ylabel(f'L{p}')
    elif args.objective == 'cosine':
        plt.ylabel(f'cos')
    plt.savefig(log_dir / f'loss_plot.png')
