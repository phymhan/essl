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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters_inv', type=int, default=500)
    parser.add_argument('--iters', type=int, default=4000)
    parser.add_argument('--lr_inv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_root', type=str, default='logs_new')
    parser.add_argument('--name', type=str, default='test')
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

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    index = np.random.choice(range(50000), size=args.batch_size, replace=False)

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

    with torch.no_grad():
        latent = generator.get_latent(torch.randn(8*8, latent_dim, device=device))
        imgs_gen, _ = generator([latent],
                                truncation=truncation,
                                truncation_latent=trunc,
                                input_is_latent=True,
                                randomize_noise=True)

        result = []
        for row in imgs_gen.chunk(8, dim=0):
            result.append(torch.cat([img for img in row], dim=2))
        result = torch.cat(result, dim=1)
        print('generated samples:')
        imsave(log_dir / 'gen_samples.png', tensor2image(result))

    # Domain-guided encoder
    # In-Domain Images
    batch_size = args.batch_size

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # dataset = datasets.CIFAR10(root=args.data_root, download=True, transform=transform)
    # loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))
    import pickle

    with open('data/c10_data2.pkl', 'rb') as f:
        data = pickle.load(f)
    images = data['images']
    labels = data['labels']
    print('[data loaded]')
    with open('data/c10_latent.pkl', 'rb') as f:
        latents = pickle.load(f)['latents']
    print('[latents loaded]')

    # imgs, _ = next(loader)
    # imgs = imgs.to(device)

    imgs = images[index].clone().cuda()
    z1 = latents[index].clone().cuda()

    imgs_real = torch.cat([img for img in imgs], dim=1)

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    
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

    h_img = model(encoder_input_transform(imgs_rep))  # precompute
    # h_img = model(encoder_input_transform(imgs_rec).repeat_interleave(n, dim=0))  # start from the reconstructed images
    h_img = normalize(h_img.squeeze()).detach()

    losses = []
    print('generating views...')
    done = False
    mask = (1-torch.eye(n)).to(device).unsqueeze(0).repeat(batch_size, 1, 1).bool()
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
        # pdist = pdist[mask].view(batch_size, n, n-1).min(dim=2)[0]
        # loss_reg = torch.mean(F.relu(eps2 - pdist.view(-1)))

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
            # optimizer.zero_grad()
            z.grad = None
            loss.backward()
            z_delta = z.grad.data.sign()
            z = (z - args.lr * z_delta).detach()
            # done = True
            z.requires_grad = True
        
        if args.objective == 'norm':
            losses.append(torch.norm(h_gen - h_img, dim=1, p=p).mean().item())
        elif args.objective == 'cosine':
            losses.append(torch.sum(h_gen * h_img, dim=1).mean().item())

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

            # pdist = torch.cdist(h_gen.view(batch_size, n, -1), h_gen.view(batch_size, n, -1), p=p)
            # pdist = pdist * n / (n-1)

            if args.objective == 'norm':
                print(f'step: {step+1}, loss: {loss.item()}, norm: {torch.norm(h_gen - h_img, dim=1, p=p).mean().item()}, pdist: {pdist.mean().item()}')
            elif args.objective == 'cosine':
                print(f'step: {step+1}, loss: {loss.item()}, cos: {torch.sum(h_gen * h_img, dim=1).mean().item()}, pdist: {pdist.mean().item()}')

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
