import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision
from torchvision import datasets, transforms

from stylegan_model import Generator, Encoder

# from functools import partial
# from pathlib import Path
import math
import pdb
st = pdb.set_trace

def toggle_grad(model, on_or_off):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = on_or_off

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss

def get_gan_models(device, no_encoder=False, g_ckpt='../pretrained/stylegan2-c10_g.pt'):
    print('loading stylegan models...')
    # g_model_path = '../pretrained/stylegan2-c10_g.pt'
    g_model_path = g_ckpt
    g_ckpt = torch.load(g_model_path, map_location=device)
    latent_dim = g_ckpt['args'].latent
    generator = Generator(32, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')
    if no_encoder:
        encoder = None
    else:
        e_model_path = '../pretrained/stylegan2-c10_e.pt'
        e_ckpt = torch.load(e_model_path, map_location=device)
        encoder = Encoder(32, latent_dim).to(device)
        encoder.load_state_dict(e_ckpt['e'])
        encoder.eval()
        print('[encoder loaded]')
    return generator, encoder


class StyleGANGenerator(nn.Module):
    def __init__(
        self,
        gan_generator=None,
        gan_encoder=None,
        simclr_encoder=None,
        idinvert_steps=100,
        boundary_steps=100,
        boundary_epsilon=0.9,
        fgsm_stepsize=0.1,
        freeze_encoder=False,
        **kwargs
    ):
        super().__init__()
        self.gan_generator = gan_generator
        self.gan_encoder = gan_encoder
        self.simclr_encoder = simclr_encoder
        # self.idinvert_steps = idinvert_steps
        self.boundary_steps = boundary_steps
        self.boundary_epsilon = boundary_epsilon
        trunc = self.gan_generator.mean_latent(4096).detach().clone()
        self.register_buffer('gan_trunc', trunc)
        self.vgg_loss = VGGLoss('cuda')
        self.input2simclr_transform = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        self.gan2simclr_transform = T.Compose([
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])  # gan to simclr default
        self.input2gan_transform = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0, 1] to [-1, 1]
        self.gan2output_transform = T.Normalize([-1, -1, -1], [2, 2, 2])  # [-1, 1] to [0, 1]
        toggle_grad(self.gan_generator, False)
        toggle_grad(self.gan_encoder, False)
        if freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        self.freeze_encoder = freeze_encoder
        self.fgsm_stepsize = fgsm_stepsize

    def forward(self, x, w=None):
        # invert x
        if w is None:
            xx = self.input2gan_transform(x)
            w0 = self.gan_encoder(xx)
            w1 = w0.detach().clone() + torch.randn_like(w0) * 0.001
        else:
            w1 = w.detach().clone()
        x1 = x
        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        w = w1
        w.requires_grad = True
        z_x1 = self.simclr_encoder(self.input2simclr_transform(x1))
        z_x1 = F.normalize(z_x1, dim=1)
        x_gen, _ = self.gan_generator(
            [w],
            input_is_latent=True,
            truncation=0.7,
            truncation_latent=self.gan_trunc,
            randomize_noise=False)
        z_gen = self.simclr_encoder(self.gan2simclr_transform(x_gen))
        z_gen = F.normalize(z_gen, dim=1)
        loss = torch.mean(F.relu(torch.sum(z_gen * z_x1.detach(), dim=1) - self.boundary_epsilon))
        loss.backward()
        with torch.no_grad():
            w_delta = w.grad.data.sign()
            w = w - self.fgsm_stepsize * w_delta
            x_gen, _ = self.gan_generator(
                [w],
                input_is_latent=True,
                truncation=0.7,
                truncation_latent=self.gan_trunc,
                randomize_noise=False)
        x_gen = self.gan2output_transform(torch.clamp(x_gen, -1, 1))
        x_gen = x_gen.detach().clone()
        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, True)
        return x_gen


class SpatialTransformerGenerator(nn.Module):
    def __init__(
        self,
        simclr_encoder=None,
        fgsm_stepsize=0.1,
        freeze_encoder=False,
        boundary_epsilon=0.6,
        **kwargs
    ):
        super().__init__()
        self.simclr_encoder = simclr_encoder
        self.input2simclr_transform = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        if freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        self.freeze_encoder = freeze_encoder
        self.fgsm_stepsize = fgsm_stepsize
        self.boundary_epsilon = boundary_epsilon
    
    def forward(self, x, w=None, **kwargs):
        init_scale = kwargs.get('init_scale', 1)

        b = x.size(0)
        theta = torch.zeros(b, 6, device=x.device)
        theta[:, 0] = init_scale
        theta[:, 4] = init_scale
        theta += torch.randn_like(theta) * 0.005  # NOTE: add hardcoded randomness

        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        theta.requires_grad = True
        grid = F.affine_grid(theta.view(b, 2, 3), x.size())
        x_gen = F.grid_sample(x, grid)

        z_x = self.simclr_encoder(self.input2simclr_transform(x))
        z_x = F.normalize(z_x, dim=1)
        z_gen = self.simclr_encoder(self.input2simclr_transform(x_gen))
        z_gen = F.normalize(z_gen, dim=1)
        loss = torch.mean(F.relu(torch.sum(z_gen * z_x.detach(), dim=1) - self.boundary_epsilon))
        loss.backward()
        with torch.no_grad():
            theta_delta = theta.grad.data.sign()
            theta = theta - self.fgsm_stepsize * theta_delta
            grid = F.affine_grid(theta.view(b, 2, 3), x.size())
            x_new = F.grid_sample(x, grid)
        x_new = x_new.detach().clone()
        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, True)
        return x_new


class RandomSpatialTransformerGenerator(nn.Module):
    def __init__(
        self,
        simclr_encoder=None,
        **kwargs,
    ):
        super().__init__()
        self.simclr_encoder = simclr_encoder

    def forward(self, x, w=None, **kwargs):
        init_scale = kwargs.get('init_scale', 1)
        noise_std = kwargs.get('noise_std', 0.1)
        with torch.no_grad():
            b = x.size(0)
            theta = torch.zeros(b, 6, device=x.device)
            theta[:, 0] = init_scale
            theta[:, 4] = init_scale
            theta += torch.randn_like(theta) * noise_std
            grid = F.affine_grid(theta.view(b, 2, 3), x.size())
            x_new = F.grid_sample(x, grid)
        return x_new


class RandomSpatialTransformerGenerator2(nn.Module):
    """
    reproduces randomresizedcrop
    """
    def __init__(
        self,
        simclr_encoder=None,
        **kwargs,
    ):
        super().__init__()
        self.simclr_encoder = simclr_encoder

    def forward(self, x, w=None, **kwargs):
        # NOTE: strict version
        b = x.size(0)
        scale = kwargs.get('scale', [0.08, 1.0])
        ratio = kwargs.get('ratio', [3. / 4., 4. / 3.])
        noise_std = kwargs.get('noise_std', 0.02)
        # NOTE: uniform sample from log will favor smaller values, for aspect_ratio = w / h,
        #       we sample more h > w (tall images, which will be horizontally streched).
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area_ratio_sqrt = torch.sqrt(torch.empty(b).uniform_(scale[0], scale[1]))
            aspect_ratio_sqrt = torch.exp(torch.empty(b).uniform_(log_ratio[0], log_ratio[1]) / 2)
            scale_x = target_area_ratio_sqrt * aspect_ratio_sqrt
            scale_y = target_area_ratio_sqrt / aspect_ratio_sqrt
            # if kwargs.get('same_along_batch', False):
            #     index = torch.randint(0, b, size=(1,)).repeat(b)
            #     scale_x = scale_x[index]
            #     scale_y = scale_y[index]
            valid = torch.bitwise_and(scale_x <= 1, scale_y <= 1)
            if not kwargs.get('strict', False) or valid.all():
                break
            elif valid.any():
                index = np.where(valid.cpu().numpy())[0]
                index_ = np.random.choice(index, b - valid.sum().item())
                index = np.concatenate([index, index_])
                target_area_ratio_sqrt = target_area_ratio_sqrt[index]
                aspect_ratio_sqrt = aspect_ratio_sqrt[index]
                scale_x = target_area_ratio_sqrt * aspect_ratio_sqrt
                scale_y = target_area_ratio_sqrt / aspect_ratio_sqrt
                break
        if kwargs.get('clamp', False):
            scale_x = torch.clamp(scale_x, max=1)
            scale_y = torch.clamp(scale_y, max=1)
        margin_x = F.relu(1 - scale_x)
        margin_y = F.relu(1 - scale_y)
        shift_x = torch.empty(b).uniform_(-1, 1) * margin_x
        shift_y = torch.empty(b).uniform_(-1, 1) * margin_y
        with torch.no_grad():
            theta = torch.zeros(b, 6, device=x.device)
            theta[:, 0] = scale_x
            theta[:, 4] = scale_y
            theta[:, 2] = shift_x
            theta[:, 5] = shift_y
            theta += torch.randn_like(theta) * noise_std
            grid = F.affine_grid(theta.view(b, 2, 3), x.size())
            x_new = F.grid_sample(x, grid)
        return x_new


class RandomResizedCropGenerator(nn.Module):
    """
    reproduces randomresizedcrop
    """
    def __init__(
        self,
        simclr_encoder=None,
        **kwargs,
    ):
        super().__init__()
        self.simclr_encoder = simclr_encoder
        # self.transform = T.RandomResizedCrop(scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), size=32)
        # print(f"RandomResizedCrop with scale=[0.2, 1.0], ratio=[3. / 4., 4. / 3.]")

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = TF.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, x, w=None, **kwargs):
        # b = x.size(0)
        # b = 1  # NOTE: RandomResizedCrop uses the same params for the whole batch
        # width, height = TF.get_image_size(x)
        # area = width * height
        # scale = kwargs.get('scale', [0.08, 1.0])
        # ratio = kwargs.get('ratio', [3. / 4., 4. / 3.])
        # log_ratio = torch.log(torch.tensor(ratio))
        # i, j = None, None
        # for _ in range(10):
        #     target_area_sqrt = torch.sqrt(area * torch.empty(b).uniform_(scale[0], scale[1]))
        #     aspect_ratio_sqrt = torch.exp(torch.empty(b).uniform_(log_ratio[0], log_ratio[1]) / 2)
        #     w = int(torch.round(target_area_sqrt * aspect_ratio_sqrt))
        #     h = int(torch.round(target_area_sqrt / aspect_ratio_sqrt))
        #     if 0 < w <= width and 0 < h <= height:
        #         i = torch.randint(0, height - h + 1, size=(b,)).item()
        #         j = torch.randint(0, width - w + 1, size=(b,)).item()
        #         break
        # # NOTE: fallback to central crop as in original implementation
        # if i is None:
        #     in_ratio = float(width) / float(height)
        #     if in_ratio < min(ratio):
        #         w = width
        #         h = int(round(w / min(ratio)))
        #     elif in_ratio > max(ratio):
        #         h = height
        #         w = int(round(h * max(ratio)))
        #     else:  # whole image
        #         w = width
        #         h = height
        #     i = (height - h) // 2
        #     j = (width - w) // 2
        # x_new = TF.resized_crop(x, i, j, h, w, (height, width))

        # x_new = torch.stack([self.transform(x[i]) for i in range(x.size(0))])
        # return x_new
        scale = kwargs.get('scale', [0.2, 1.0])
        ratio = kwargs.get('ratio', [3. / 4., 4. / 3.])
        # x_new = []
        # for k in range(x.size(0)):
        #     i, j, h, w = self.get_params(x[k], scale, ratio)
        #     x_new.append(TF.resized_crop(x[k], i, j, h, w, x.shape[-2:]))
        # return torch.stack(x_new)
        i, j, h, w = self.get_params(x, scale, ratio)
        x_new = TF.resized_crop(x, i, j, h, w, x.shape[-2:])
        return x_new


class GaussianViewGenerator(nn.Module):
    def __init__(
        self,
        gan_generator=None,
        gan_encoder=None,
        simclr_encoder=None,
        idinvert_steps=100,
        boundary_steps=100,
        boundary_epsilon=0.9,
        fgsm_stepsize=0.1,
        freeze_encoder=False,
        **kwargs
    ):
        super().__init__()
        self.gan_generator = gan_generator
        self.gan_encoder = gan_encoder
        self.simclr_encoder = simclr_encoder
        self.boundary_steps = boundary_steps
        self.boundary_epsilon = boundary_epsilon
        trunc = self.gan_generator.mean_latent(4096).detach().clone()
        self.register_buffer('gan_trunc', trunc)
        # self.vgg_loss = VGGLoss('cuda')
        self.input2simclr_transform = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        self.gan2simclr_transform = T.Compose([
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])  # gan to simclr default
        self.input2gan_transform = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0, 1] to [-1, 1]
        self.gan2output_transform = T.Normalize([-1, -1, -1], [2, 2, 2])  # [-1, 1] to [0, 1]
        toggle_grad(self.gan_generator, False)
        toggle_grad(self.gan_encoder, False)
        if freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        self.freeze_encoder = freeze_encoder
        self.fgsm_stepsize = fgsm_stepsize

    def forward(self, x, w=None, **kwargs):
        noise_std = kwargs.get('noise_std', 0.2)
        w_noise = torch.randn_like(w) * noise_std
        w = w + w_noise
        with torch.no_grad():
            x_gen, _ = self.gan_generator(
                [w],
                input_is_latent=True,
                truncation=0.7,
                truncation_latent=self.gan_trunc,
                randomize_noise=False)
        x_gen = self.gan2output_transform(torch.clamp(x_gen, -1, 1))
        x_gen = x_gen.detach().clone()
        return x_gen
