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
from main import ContrastiveLearningTransform2, Branch
# from main2 import ContrastiveLearningTransform, Branch

from functools import partial
from tqdm import tqdm
import pdb
st = pdb.set_trace


class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='data/c10_data2.pkl')
    parser.add_argument('--latent_path', type=str, default='data/c10_latent.pkl')
    parser.add_argument('--view_path', type=str, default='data/c10_obj=l2_eps=0.3_loss=l2_reg=0.01_eps2=0.5_n=8_seed=0.pkl')
    args = parser.parse_args()

    device = 'cuda'

    # model = Branch(args_simclr).to(device)
    # saved_dict = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')['state_dict']
    # model.load_state_dict(saved_dict, strict=True)
    opt = args
    if opt.dataset == 'cifar10':
        from main import Branch
        args_simclr = Config(dim_proj='2048,2048', dim_pred=512, loss='simclr')
        model = Branch(args_simclr).to(device)
        saved_dict = torch.load('../pretrained/simclr-cifar10-resnet18-800ep-1.pth')['state_dict']
        model.load_state_dict(saved_dict, strict=True)
        # expert_transform = ContrastiveLearningTransformCIFAR10()
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
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
        # expert_transform = ContrastiveLearningTransformCIFAR100()
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tiny':
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
        # expert_transform = ContrastiveLearningTransformTinyImageNet1()
        mean = (0.480, 0.448, 0.398)
        std = (0.277, 0.269, 0.282)
    
    normalize = T.Normalize(mean=mean, std=std)
    # single_transform = T.Compose([T.ToTensor(), normalize])
    single_transform = T.Compose([T.ToTensor(), normalize])
    
    model.eval()  # NOTE

    batch_size = 256

    from utils_data import MultiViewDataset
    train_dataset = MultiViewDataset(
        data_path=args.data_path,
        latent_path=args.latent_path,
        view_paths=[args.view_path],
        transform0=normalize,
        transform1=normalize,
        transform2=normalize,
        transform3=normalize,
        n_views=2,  # NOTE: we use 0, 3, 4
        train=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
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

    loader = sample_data(train_loader)

    with torch.no_grad():
        for i in tqdm(range(195)):
            imgs, _, _ = next(loader)
            h0 = model(imgs[0].cuda())
            h1 = model(imgs[3].cuda())
            h2 = model(imgs[4].cuda())
            h0 = F.normalize(h0, dim=-1, p=2)
            h1 = F.normalize(h1, dim=-1, p=2)
            h2 = F.normalize(h2, dim=-1, p=2)
            
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
    
