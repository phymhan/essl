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

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import utils
import pdb
st = pdb.set_trace

def toggle_grad(model, on_or_off):
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

def get_gan_models(device):
    print('loading stylegan models...')
    g_model_path = '../pretrained/stylegan2-c10_g.pt'
    g_ckpt = torch.load(g_model_path, map_location=device)
    latent_dim = g_ckpt['args'].latent
    generator = Generator(32, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')
    e_model_path = '../pretrained/stylegan2-c10_e.pt'
    e_ckpt = torch.load(e_model_path, map_location=device)
    encoder = Encoder(32, latent_dim).to(device)
    encoder.load_state_dict(e_ckpt['e'])
    encoder.eval()
    print('[encoder loaded]')
    return generator, encoder


class ViewGenerator(nn.Module):
    def __init__(
        self,
        gan_generator=None,
        gan_encoder=None,
        simclr_encoder=None,
        idinvert_steps=100,
        boundary_steps=100,
        boundary_epsilon=0.9,
        **kwargs
    ):
        super().__init__()
        self.gan_generator = gan_generator
        self.gan_encoder = gan_encoder
        self.simclr_encoder = simclr_encoder
        self.idinvert_steps = idinvert_steps
        self.boundary_steps = boundary_steps
        self.boundary_epsilon = boundary_epsilon
        trunc = self.gan_generator.mean_latent(4096).detach().clone()
        self.register_buffer('gan_trunc', trunc)
        self.vgg_loss = VGGLoss('cuda')
        self.gan_input_transform = T.Compose(
            [
                T.Normalize([-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], [1/0.2023, 1/0.1994, 1/0.2010]),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )  # to [-1, 1]
        self.gan_output_transform = T.Compose(
            [
                T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )  # to simclr default
        toggle_grad(self.gan_generator, False)
        toggle_grad(self.gan_encoder, False)

    def forward(self, x):
        # invert x
        xx = self.gan_input_transform(x)
        w0 = self.gan_encoder(xx)
        
        # idinvert
        if self.idinvert_steps > 0:
            w = w0.detach().clone()
            w.requires_grad = True
            w_optim = torch.optim.Adam([w], lr=0.01)
            for i in range(self.idinvert_steps):
                x_rec, _ = self.gan_generator(
                    [w],
                    input_is_latent=True,
                    truncation=0.7,
                    truncation_latent=self.gan_trunc,
                    randomize_noise=False)
                w_hat = self.gan_encoder(x_rec)
                loss = F.mse_loss(x_rec, xx) + self.vgg_loss(x_rec, xx) + F.mse_loss(w0, w_hat) * 2.0
                w_optim.zero_grad()
                loss.backward()
                w_optim.step()
            w1 = w.detach().clone()
        else:
            w1 = w0.detach().clone()

        x_rec, _ = self.gan_generator(
            [w1],
            input_is_latent=True,
            truncation=0.7,
            truncation_latent=self.gan_trunc,
            randomize_noise=False)
        x1 = self.gan_output_transform(x_rec)

        # boundary
        if self.boundary_steps > 0:
            w = w1
            w.requires_grad = True
            w_optim = torch.optim.Adam([w], lr=0.01)
            with torch.no_grad():
                # z_x1 = self.simclr_encoder(x)
                z_x1 = self.simclr_encoder(x1)
                z_x1 = F.normalize(z_x1, dim=1)
            for i in range(self.boundary_steps):
                x_gen, _ = self.gan_generator(
                    [w],
                    input_is_latent=True,
                    truncation=0.7,
                    truncation_latent=self.gan_trunc,
                    randomize_noise=False)
                # TODO: how to speed up simclr encoder forward?
                z_gen = self.simclr_encoder(self.gan_output_transform(x_gen))
                z_gen = F.normalize(z_gen, dim=1)
                loss = torch.mean(F.relu(torch.sum(z_gen * z_x1, dim=1) - self.boundary_epsilon))
                if loss.item() > 1e-5:
                    w_optim.zero_grad()
                    loss.backward()
                    w_optim.step()
                else:
                    break
            x_gen = self.gan_output_transform(x_gen)
        else:
            x_gen = x1

        return x_gen.detach()


class ViewGeneratorFGSM(nn.Module):
    def __init__(
        self,
        gan_generator=None,
        gan_encoder=None,
        simclr_encoder=None,
        idinvert_steps=100,
        boundary_steps=100,
        boundary_epsilon=0.9,
        fgsm_eps=0.1,
        freeze_encoder=False,
        **kwargs
    ):
        super().__init__()
        self.gan_generator = gan_generator
        self.gan_encoder = gan_encoder
        self.simclr_encoder = simclr_encoder
        self.idinvert_steps = idinvert_steps
        self.boundary_steps = boundary_steps
        self.boundary_epsilon = boundary_epsilon
        trunc = self.gan_generator.mean_latent(4096).detach().clone()
        self.register_buffer('gan_trunc', trunc)
        self.vgg_loss = VGGLoss('cuda')
        self.gan_input_transform = T.Compose(
            [
                T.Normalize([-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], [1/0.2023, 1/0.1994, 1/0.2010]),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )  # to [-1, 1]
        self.gan_output_transform = T.Compose(
            [
                T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )  # to simclr default
        toggle_grad(self.gan_generator, False)
        toggle_grad(self.gan_encoder, False)
        if freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        self.freeze_encoder = freeze_encoder
        self.fgsm_eps = fgsm_eps

    def forward(self, x):
        # invert x
        xx = self.gan_input_transform(x)
        w0 = self.gan_encoder(xx)
        w1 = w0.detach().clone() + torch.randn_like(w0) * 0.001
        x1 = x

        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, False)
        w = w1
        w.requires_grad = True
        z_x1 = self.simclr_encoder(x1)
        z_x1 = F.normalize(z_x1, dim=1)
        x_gen, _ = self.gan_generator(
            [w],
            input_is_latent=True,
            truncation=0.7,
            truncation_latent=self.gan_trunc,
            randomize_noise=False)
        z_gen = self.simclr_encoder(self.gan_output_transform(x_gen))
        z_gen = F.normalize(z_gen, dim=1)
        loss = torch.mean(F.relu(torch.sum(z_gen * z_x1.detach(), dim=1) - self.boundary_epsilon))
        loss.backward()
        with torch.no_grad():
            w_delta = w.grad.data.sign()
            w = w - self.fgsm_eps * w_delta
            x_gen, _ = self.gan_generator(
                [w],
                input_is_latent=True,
                truncation=0.7,
                truncation_latent=self.gan_trunc,
                randomize_noise=False)
            x_gen = self.gan_output_transform(torch.clamp(x_gen, -1, 1))
            x_gen = x_gen.detach().clone()
        if not self.freeze_encoder:
            toggle_grad(self.simclr_encoder, True)
        return x_gen
