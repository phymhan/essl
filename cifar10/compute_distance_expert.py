from ast import Mult
import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import torchvision.transforms as T

# from main2 import ContrastiveLearningTransform, Branch

from functools import partial
from tqdm import tqdm
import pdb
st = pdb.set_trace


class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class ContrastiveLearningTransformCIFAR10:
    def __init__(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        output = [
            self.single_transform(x),
            self.transform(x),
            self.transform(x),
        ]
        return output

class ContrastiveLearningTransformCIFAR100:
    def __init__(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        output = [
            self.single_transform(x),
            self.transform(x),
            self.transform(x),
        ]
        return output

class ContrastiveLearningTransformTinyImageNet1:
    def __init__(self):
        mean = (0.480, 0.448, 0.398)
        std = (0.277, 0.269, 0.282)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.transform = T.Compose([
            T.RandomResizedCrop(size=64, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        output = [
            self.single_transform(x),
            self.transform(x),
            self.transform(x),
        ]
        return output

class ContrastiveLearningTransformTinyImageNet2:
    def __init__(self):
        mean = (0.480, 0.448, 0.398)
        std = (0.277, 0.269, 0.282)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.transform = T.Compose(
            [
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                T.RandomGrayscale(p=0.1),
                T.RandomResizedCrop(
                    64,
                    scale=(0.2, 1.0),
                    ratio=(0.75, 4/3.),
                    interpolation=3,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, x):
        output = [
            self.single_transform(x),
            self.transform(x),
            self.transform(x),
        ]
        return output


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_folder', type=str, default='../data')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--no_eval', action='store_true')
    # parser.add_argument('--view_path', type=str, default='data/c10_obj=l2_eps=0.3_loss=l2_reg=0.01_eps2=0.5_n=8_seed=0.pkl')
    args = parser.parse_args()

    device = 'cuda'

    from utils import fix_seed
    fix_seed(0)

    opt = args

    if opt.dataset == 'cifar10':
        from main import Branch
        args_simclr = Config(dim_proj='2048,2048', dim_pred=512, loss='simclr')
        model = Branch(args_simclr).to(device)
        saved_dict = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')['state_dict']
        model.load_state_dict(saved_dict, strict=True)
        expert_transform = ContrastiveLearningTransformCIFAR10()
    elif opt.dataset == 'cifar100':
        from resnet_big import SupConResNet
        # import apex
        model = SupConResNet(name='resnet50')
        # model = apex.parallel.convert_syncbn_model(model)
        # model.encoder = torch.nn.DataParallel(model.encoder)
        state_dict = torch.load('../../SupContrast/logs/0425_c100_bs=1024_base/weights/last.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        expert_transform = ContrastiveLearningTransformCIFAR100()
    elif opt.dataset == 'tiny1':
        from resnet_big import SupConResNet
        # supcon codebase
        model = SupConResNet(name='resnet50')
        state_dict = torch.load('../../SupContrast/logs/0430_t64_bs=512_base_2/weights/ckpt_epoch_1000.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        expert_transform = ContrastiveLearningTransformTinyImageNet1()
    
    elif opt.dataset == 'tiny2':
        # self-supervised codebase
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
        # model = model.model
        expert_transform = ContrastiveLearningTransformTinyImageNet2()
    
    if not args.no_eval:
        model.eval()  # NOTE

    batch_size = args.batch_size

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=expert_transform,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=expert_transform,
                                          download=True)
    elif opt.dataset == 'tiny1' or opt.dataset == 'tiny2':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                            transform=expert_transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )

    p = 2
    n_views = 3

    d01_list = list()
    d02_list = list()
    d12_list = list()

    s01_list = list()
    s02_list = list()
    s12_list = list()

    s_pos = list()
    s_neg = list()
    d_pos = list()
    d_neg = list()

    # loader = sample_data(train_loader)

    with torch.no_grad():
        # for i in tqdm(range(195)):
        for batch in tqdm(train_loader):
            # imgs, _ = next(loader)
            imgs = batch[0]

            h0 = model(imgs[0].cuda())
            h1 = model(imgs[1].cuda())
            h2 = model(imgs[2].cuda())
            h0 = F.normalize(h0, dim=1)
            h1 = F.normalize(h1, dim=1)
            h2 = F.normalize(h2, dim=1)
            
            d01 = torch.norm(h0 - h1, p=p, dim=1)
            d02 = torch.norm(h0 - h2, p=p, dim=1)
            d12 = torch.norm(h1 - h2, p=p, dim=1)
            s01 = torch.sum(h0 * h1, dim=1)
            s02 = torch.sum(h0 * h2, dim=1)
            s12 = torch.sum(h1 * h2, dim=1)
            
            d01_list.append(d01)
            d02_list.append(d02)
            d12_list.append(d12)
            s01_list.append(s01)
            s02_list.append(s02)
            s12_list.append(s12)
            
            features = torch.cat([h0, h1, h2], dim=0)
            similarity_matrix = torch.matmul(features, features.T)
            distance_matrix = torch.cdist(features, features, p=p)
            
            labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            mask = torch.eye(labels.shape[0], dtype=torch.bool)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            distance_matrix = distance_matrix[~mask].view(similarity_matrix.shape[0], -1)
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
            s_pos.append(positives)
            s_neg.append(negatives)
            
            dist_pos = distance_matrix[labels.bool()].view(labels.shape[0], -1)
            dist_neg = distance_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
            d_pos.append(dist_pos)
            d_neg.append(dist_neg)

    d01 = torch.cat(d01_list, dim=0)
    d02 = torch.cat(d02_list, dim=0)
    d12 = torch.cat(d12_list, dim=0)
    s01 = torch.cat(s01_list, dim=0)
    s02 = torch.cat(s02_list, dim=0)
    s12 = torch.cat(s12_list, dim=0)
    s_pos = torch.cat(s_pos, dim=0)
    s_neg = torch.cat(s_neg, dim=0)
    d_pos = torch.cat(d_pos, dim=0)
    d_neg = torch.cat(d_neg, dim=0)
    print(f"d01: {d01.mean().item():.4f} +- {d01.std().item():.4f}")
    print(f"distance: (0, 1): {d01.mean().item():.4f}, (0, 2): {d02.mean().item():.4f}, (1, 2): {d12.mean().item():.4f}")
    print(f"cosine: (0, 1): {s01.mean().item():.4f}, (0, 2): {s02.mean().item():.4f}, (1, 2): {s12.mean().item():.4f}")
    print(f"cosine: positive: {s_pos.mean().item():.4f}, negative: {s_neg.mean().item():.4f}")
    print(f"distance: positive: {d_pos.mean().item():.4f}, negative: {d_neg.mean().item():.4f}")

    # save visuals
    
