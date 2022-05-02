import io
import sys
import os
import sys
import requests
import PIL
import warnings
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt, log
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# from dalle_pytorch import distributed_utils
import torchvision
import pdb

st = pdb.set_trace


class VQGanVAE1024(nn.Module):
    def __init__(self, vae_path=None, image_size=None):
        super().__init__()

        model_filename = 'vqgan.1024.model.ckpt'
        config_filename = 'vqgan.1024.config.yml'

        config_path = str(Path('pretrained') / config_filename)
        config = OmegaConf.load(config_path)
        if image_size:
            config.model.params['ddconfig']['resolution'] = image_size
        model = VQModel(**config.model.params)

        if vae_path is None:
            model_path = str(Path('pretrained') / model_filename)
        else:
            model_path = vae_path
        state = torch.load(model_path, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=False)

        self.model = model

        self.num_layers = 4
        self.image_size = 256
        self.num_tokens = 1024

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices.squeeze(), '(b n) -> b n', b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        # one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        # z = (one_hot_indices @ self.model.quantize.embedding.weight)
        z = self.model.quantize.embedding(img_seq)

        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def decode_train(self, probs):
        # probs [B, N, D]
        b, n, d = probs.shape
        # one_hot_indices = F.one_hot(logits, num_classes = self.num_tokens).float()
        one_hot_indices = probs

        z = (one_hot_indices @ self.model.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048, 2048, 2048, 128]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector = nn.Sequential(*layers)

        self.online_head = nn.Linear(2048, 1000)

        if args.rotation:
            self.rotation_projector = nn.Sequential(nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # first layer
                                                    nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # second layer
                                                    nn.Linear(2048, 128),
                                                    nn.LayerNorm(128),
                                                    nn.Linear(128, 4))  # output layer


    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        # projoection
        z1 = self.projector(r1)
        z2 = self.projector(r2)


        loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        logits = self.online_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc

    def forward_rotation(self, x):
        b = self.backbone(x)
        logits = self.rotation_projector(b)

        return logits
