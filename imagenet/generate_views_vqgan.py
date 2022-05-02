import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torchvision import datasets, transforms

# from stylegan_model import Generator, Encoder
# from view_generator import VGGLoss

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pickle
from utils import toggle_grad, image2tensor, tensor2image, imshow, imsave, Config, fix_seed

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

class SimCLRWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def forward(self, x):
        r = self.model.backbone(x)
        z = self.model.projector(r)
        return z

def encode(vqgan, x):
    h = vqgan.encoder(x)
    z = vqgan.quant_conv(h)
    z_q, _, [_, _, indices] = vqgan.quantize(z)
    return z_q, z, indices

def decode(vqgan, z_q):
    quant = vqgan.post_quant_conv(z_q)
    x = vqgan.decoder(quant)
    return x

def decode_z(vqgan, z):
    z_q, _, info = vqgan.quantize(z)
    quant = vqgan.post_quant_conv(z_q)
    x = vqgan.decoder(quant)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--iters', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_root', type=str, default='logs_gen')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=1)
    parser.add_argument('--init_noise_scale', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--lam2', type=float, default=0, help='weight of distance regularization')
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--objective', type=str, default='norm', choices=['norm', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l2', 'l1', 'hinge'])
    parser.add_argument('--method', type=str, default='gd', choices=['gd', 'fgsm', 'vat'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--which_decode', type=str, default='from_z', choices=['from_z', 'from_zq'])
    parser.add_argument('--optimize_dict', action='store_true')

    args = parser.parse_args()

    fix_seed(args.seed)

    name = args.name
    log_root = Path(args.log_root)
    log_dir = log_root / name
    os.makedirs(log_dir, exist_ok=True)

    USE_HTML = True
    log_web_dir = log_dir / 'web'
    webpage = None
    if USE_HTML:
        import utils_html
        webpage = utils_html.initialize_webpage(log_web_dir, 'ViewMaker: ' + args.name + f'{args.which_decode}', resume=False)

    device = 'cuda'
    image_size = args.image_size
    tol = 1e-5
    image_size = 256
    args.clamp = True

    vqgan_config_path = '/research/cbim/medical/lh599/code/DALLE/pretrained/vqgan.1024.config.yml'
    vqgan_model_path = '/research/cbim/medical/lh599/code/DALLE/pretrained/vqgan.1024.model.ckpt'
    
    from taming.models.vqgan import VQModel
    from omegaconf import OmegaConf
    config = OmegaConf.load(vqgan_config_path)
    config.model.params['ddconfig']['resolution'] = image_size
    vqgan = VQModel(**config.model.params)
    state = torch.load(vqgan_model_path, map_location='cpu')['state_dict']
    vqgan.load_state_dict(state, strict=True)  # NOTE: set False if resoluton is changed

    # Define SimCLR encoder
    if args.no_proj:
        exit(0)
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
        # from simclr.main import SimCLR
        from models import SimCLR
        normalize = partial(F.normalize, dim=1)
        prefix = 'proj'
        args_simclr = Config(rotation=0)
        model = SimCLR(args_simclr).to(device)
        saved_dict = torch.load('../pretrained/simclr-imagenet-resnet50-300ep.pth')

        model.backbone.load_state_dict(saved_dict['backbone'])
        model.projector.load_state_dict(saved_dict['projector'])
        # model.load_state_dict(saved_dict['model'], strict=True)
        model = SimCLRWrapper(model)
    
    model.eval()

    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_cache = 'I256_256.pkl'
    if not os.path.exists(image_cache):
        # dataset = torchvision.datasets.ImageFolder(root='/research/cbim/medical/lh599/data/ILSVRC2012/train', transform=transform)
        dataset = torchvision.datasets.ImageFolder(root='/research/cbim/medical/lh599/data/ImageNet100/train', transform=transform)
        loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True))
        imgs, _ = next(loader)
        with open(image_cache, 'wb') as f:
            pickle.dump(imgs, f)
    else:
        with open(image_cache, 'rb') as f:
            imgs = pickle.load(f)
    imgs = imgs[:batch_size].to(device)
    vqgan = vqgan.to(device)

    with torch.no_grad():
        z0_q, z0, indices = encode(vqgan, imgs)
        imgs_gen = decode(vqgan, z0_q)

    # NOTE: z and z_q are of shape [b, 256, 16, 16]
    imgs_real = torch.cat([img for img in imgs], dim=1)
    imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
    imsave(log_dir / f'rec.png', tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)))

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Resize(224),  # TODO: is this backpropable? -> yes
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eps1 = args.eps1
    eps2 = args.eps2
    p = args.p
    n = args.n

    z = z0_q.detach().clone()
    z = z.repeat_interleave(n, dim=0)

    z_noise = args.init_noise_scale * torch.randn_like(z)  # [b*n, L, D]
    z_noise[::n, ...] = 0  # make sure the first one is accurate
    z = z + z_noise

    imgs_rep = imgs.repeat_interleave(n, dim=0)

    # decode_fn = decode_z  # TODO: test out different decode
    if args.which_decode == 'from_z':
        decode_fn = decode_z
    elif args.which_decode == 'from_zq':
        decode_fn = decode

    imgs_rec = imgs_gen.detach()
    if args.clamp:
        imgs_rec = torch.clamp(imgs_rec, -1, 1)
    imgs_recon = imgs_fakes
    imgs_blank = torch.ones_like(imgs_recon)[:,:,:8]

    z.requires_grad = True
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([z], lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([z], lr=args.lr)
    elif args.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS([z], max_iter=500)  # TODO: max_iter

    toggle_grad(model, False)
    toggle_grad(vqgan, False)

    h_img = model(encoder_input_transform(imgs_rec).repeat_interleave(n, dim=0))  # start from the reconstructed images
    h_img = normalize(h_img.squeeze()).detach()

    losses = []
    print('generating views...')
    done = False
    for step in range(args.iters):
        imgs_gen = decode_fn(vqgan, z)
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

        if done or step == 0 or step == args.iters - 1 or (step + 1) % args.save_every == 0:
            imgs_gen = decode_fn(vqgan, z)
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
