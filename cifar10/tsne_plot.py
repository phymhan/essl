import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torchvision import datasets, transforms

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import utils

from typing import Type, Any, Callable, Union, List, Optional
import pdb
st = pdb.set_trace

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    return
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def imsave(name, img, size=5, cmap='jet'):
    plt.imsave(name, img, cmap=cmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--log_root', type=str, default='logs_gen')
    parser.add_argument('--name', type=str, default='tsne')
    parser.add_argument('--model_path', type=str, default='../pretrained/simclr-cifar10-resnet18-800ep-1.pth')
    parser.add_argument('--data_root', type=str, default='../data')
    # parser.add_argument('--load_model', type=str, default='simclr')
    parser.add_argument('--eps1', type=float, default=0.5)
    parser.add_argument('--eps2', type=float, default=1)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--no_proj', action='store_true')
    parser.add_argument('--load_old_checkpoint', action='store_true')

    args = parser.parse_args()

    utils.fix_seed(args.seed)

    name = args.name
    log_root = Path(args.log_root)
    log_dir = log_root / name
    os.makedirs(log_dir, exist_ok=True)

    log_web_dir = log_dir / 'web'

    device = 'cuda'
    image_size = args.image_size

    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(root=args.data_root, download=True, transform=transform)
    loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))

    # do one batch
    imgs, labels = next(loader)
    imgs = imgs.to(device)

    # input transform
    encoder_input_transform = T.Compose(
        [
            T.Normalize([-1, -1, -1], [2, 2, 2]),  # to [0, 1]
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    # Define SimCLR encoder
    if args.no_proj:
        normalize = lambda x: x
        prefix = 'noproj'
        from resnet import resnet18
        model = resnet18(pretrained=False, num_classes=10)
        checkpoint = torch.load(args.model_path)
        # state_dict = checkpoint['state_dict']
        if args.load_old_checkpoint:
            saved_dict = torch.load(args.model_path)['state_dict']
        else:
            saved_dict = torch.load(args.model_path)['model_state_dict']
        for k in list(saved_dict.keys()):
            if k.startswith('encoder.'):
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    saved_dict[k[len("encoder."):]] = saved_dict[k]
            del saved_dict[k]
        log = model.load_state_dict(saved_dict, strict=True)
        # assert log.missing_keys == ['fc.weight', 'fc.bias']
        # model = ResNetWrapper(model).to(device)
        model.to(device)
    else:
        from main import Branch
        normalize = partial(F.normalize, dim=1)
        prefix = 'proj'
        args_simclr = Config(dim_proj='2048,2048', dim_pred=512, loss='simclr')
        model = Branch(args_simclr).to(device)
        if args.load_old_checkpoint:
            saved_dict = torch.load(args.model_path)['state_dict']
        else:
            saved_dict = torch.load(args.model_path)['model_state_dict']
        model.load_state_dict(saved_dict, strict=True)

    from sklearn.manifold import TSNE
    import seaborn as sns
    tsne = TSNE()

    with torch.no_grad():
        h_imgs = model(encoder_input_transform(imgs))
        h_imgs = normalize(h_imgs.squeeze())

    h_tsne = tsne.fit_transform(h_imgs.cpu().numpy())
    fig = plt.figure(figsize = (10, 10))
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(h_tsne[:,0], h_tsne[:,1], hue=labels.cpu().numpy(), legend='full', palette=sns.color_palette("bright", 10))
    plt.legend(['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'])
    plt.savefig(log_dir / f'tsne-{prefix}_{args.batch_size}.png')
    plt.close(fig)
