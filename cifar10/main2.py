import os
import math
import time
import copy
import argparse

import numpy as np
import random
import shutil
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T

from utils import knn_monitor, fix_seed, setup_wandb, print_args, get_last_checkpoint, logging_file, str2list, toggle_grad
# from losses import info_nce_loss

import pdb
st = pdb.set_trace


# normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
# normalize_inv = T.Normalize([-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], [1/0.2023, 1/0.1994, 1/0.2010])
# single_transform = T.Compose([T.ToTensor(), normalize])

NO_LOGGING = False  # NOTE: flag to turn off logging for evaluation


class BatchwiseTransform:
    def __init__(self, transform):
        # perform random transform along batch dimension
        self.transform = transform

    def __call__(self, x):
        # x: [B, C, H, W]
        y = [self.transform(i) for i in x]
        return torch.stack(y, dim=0)
    
    def __repr__(self):
        return f'BatchwiseTransform({self.transform})'

def get_default_transform(which_dataset):
    # global normalize, normalize_inv, single_transform
    if which_dataset == 'cifar10':
        normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        normalize_inv = T.Normalize([-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], [1/0.2023, 1/0.1994, 1/0.2010])
        single_transform = T.Compose([T.ToTensor(), normalize])
    elif which_dataset == 'cifar100':
        normalize = T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        normalize_inv = T.Normalize([-0.5071/0.2675, -0.4867/0.2565, -0.4408/0.2761], [1/0.2675, 1/0.2565, 1/0.2761])
        single_transform = T.Compose([T.ToTensor(), normalize])
    else:
        raise NotImplementedError
    return normalize, normalize_inv, single_transform

def get_default_dataset(which_dataset, train=True, which_transform=None):
    _, _, single_transform = get_default_transform(which_dataset)
    if which_dataset == 'cifar10':
        transform = which_transform or single_transform
        dataset = torchvision.datasets.CIFAR10(
            '../data', train=train, transform=transform, download=True
        )
    elif which_dataset == 'cifar100':
        # transform = which_transform or single_transform_c100
        transform = which_transform or single_transform
        dataset = torchvision.datasets.CIFAR100(
            '../data', train=train, transform=transform, download=True
        )
    else:
        raise NotImplementedError
    return dataset

def get_transform_list(which_transform, size=32):
    if isinstance(which_transform, str):
        which_transform = which_transform.replace(',', '+').split('+')
    which_transform = [t.lower() for t in which_transform if t]
    transform_list = []
    for t in which_transform:
        t = t.lower()
        if t == 'resizedcrop':
            transform_list.append(T.RandomResizedCrop(size=size, scale=(0.2, 1.0)))
        elif t == 'resizedcrophalf':
            transform_list.append(T.RandomResizedCrop(size=size//2, scale=(0.2, 1.0)))
        elif t == 'smallcrop':
            transform_list.append(T.RandomResizedCrop(size=size, scale=(0.9, 1.), ratio=(0.9, 1.1)))
        elif t == 'medcrop':
            transform_list.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.), ratio=(0.9, 1.1)))
        elif t == 'largecrop':
            transform_list.append(T.RandomResizedCrop(size=size, scale=(0.5, 1.), ratio=(0.9, 1.1)))
        elif t == 'horizontalflip':
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        elif t == 'colorjitter':
            transform_list.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        elif t == 'grayscale':
            transform_list.append(T.RandomGrayscale(p=0.2))
        else:
            raise NotImplementedError
    return transform_list


def get_transforms(args, which_transform):
    """
    normalize is always appended to the end, we assume all transforms (including GAN) accept
    data in [0, 1] range.
    The function will return a pre_transform and a post_transform, pre_transform is applied
    in the dataset, and post_transform is called in the training loop after data is collected.
    If GAN is not used, post_transform is None (so that all transforms will be applied in
    dataloader, which might be more efficient).
    e.g.: 'gan+resizedcrop+horizontalflip+colorjitter+grayscale'
    """
    normalize, _, _ = get_default_transform(args.dataset)
    size = args.image_size
    if isinstance(which_transform, str):
        which_transform = which_transform.replace(',', '+').split('+')
    if 'gan' in which_transform:
        index = which_transform.index('gan')
        pre_transform = get_transform_list(which_transform[:index], size=size)
        post_transform = get_transform_list(which_transform[index + 1:], size=size) + [normalize]
    else:
        pre_transform = get_transform_list(which_transform, size=size) + [normalize]
        post_transform = None
    if pre_transform is not None:
        pre_transform = T.Compose(pre_transform)
    if post_transform is not None:
        # NOTE: post_transform is called in the training loop after data is collected, so we make it batchwise
        # post_transform = [BatchwiseTransform(t) for t in post_transform]
        post_transform = BatchwiseTransform(T.Compose(post_transform))
    return pre_transform, post_transform


def get_train_dataset_and_transforms(args):
    # transforms for view_0
    pre_transform0, post_transform0 = get_transforms(args, args.which_transform0)
    assert post_transform0 is None, 'post_transform0 should be None'

    # transforms for view_1
    pre_transform1, post_transform1 = get_transforms(args, args.which_transform1)
    assert post_transform1 is None, 'post_transform1 should be None'

    # transforms for view_2
    pre_transform2, post_transform2 = get_transforms(args, args.which_transform2)
    assert post_transform2 is None, 'post_transform2 should be None'

    # transforms for view_3 and ...
    pre_transform3, post_transform3 = get_transforms(args, args.which_transform3)

    # dataset for all views
    sample_from_mixed = args.sample_from_mixed
    if args.use_pos_neg_view:
        from utils_data import PosNegViewDataset
        dataset = PosNegViewDataset(
            args.data_path,
            args.latent_path,
            pos_view_paths=args.pos_view_paths,
            neg_view_paths=args.neg_view_paths,
            transform0=pre_transform0,
            transform1=pre_transform1,
            transform2=pre_transform2,
            transform3=pre_transform3,
            n_views=args.n_views_gan,
            train=True,
            sample_from_mixed=sample_from_mixed,
            sample_original=args.sample_original,
        )
    else:
        from utils_data import MultiViewDataset
        dataset = MultiViewDataset(
            args.data_path,
            args.latent_path,
            args.view_paths,
            transform0=pre_transform0,
            transform1=pre_transform1,
            transform2=pre_transform2,
            transform3=pre_transform3,
            n_views=args.n_views_gan,
            n_cache=args.n_cache_gan,
            train=True,
            sample_from_mixed=sample_from_mixed,
            sample_original=args.sample_original,
        )
    return dataset, post_transform3


class ContrastiveLearningTransform:
    def __init__(self, dataset='cifar10'):
        transforms = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        transforms_rotation = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        _, _, single_transform = get_default_transform(dataset)
        self.single_transform = single_transform

        self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)

    def __call__(self, x):
        output = [
            self.single_transform(self.transform(x)),
            self.single_transform(self.transform(x)),
            self.single_transform(self.transform_rotation(x))
        ]
        return output


def rotate_images(images):
    nimages = images.shape[0]
    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda()
    rot_classes = torch.zeros([n_rot_images]).long().cuda()

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages:2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages:3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes


def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def negative_cosine_similarity_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce_loss(z1, z2, temperature=0.5):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def supcon_loss(z1, z2, z3, temperature=0.5):
    labels = torch.cat([torch.arange(z1.shape[0]) for i in range(3)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()
    
    z = torch.cat([z1, z2, z3], dim=0)
    z = torch.nn.functional.normalize(z, dim=1)

    logits = z @ z.T
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    logits = logits[~mask].view(logits.shape[0], -1)
    positives = logits[labels.bool()].view(labels.shape[0], -1)
    negatives = logits[~labels.bool()].view(labels.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label0 = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
    loss0 = torch.nn.functional.cross_entropy(logits, label0)
    label1 = torch.ones(labels.shape[0], dtype=torch.long).cuda()
    loss1 = torch.nn.functional.cross_entropy(logits, label1)
    loss = (loss0 + loss1) / 2

    return loss


def supcon_neg_loss(z1, z2, z3, temperature=0.5):
    bsz = z1.shape[0]

    labels = torch.cat([torch.arange(bsz) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    z_anchor = torch.cat([z1, z2], dim=0)
    z_anchor = F.normalize(z_anchor, dim=1)

    z_contra = torch.cat([z1, z2, z3], dim=0)
    z_contra = F.normalize(z_contra, dim=1)

    logits = z_anchor @ z_contra.T
    mask = torch.cat([torch.eye(bsz * 2, dtype=torch.bool), torch.zeros(bsz * 2, bsz, dtype=torch.bool)], dim=1).cuda()
    labels = torch.cat([labels, torch.zeros(bsz * 2, bsz)], dim=1).cuda()
    labels = labels[~mask].view(bsz * 2, -1)
    logits = logits[~mask].view(bsz * 2, -1)
    positives = logits[labels.bool()].view(bsz * 2, -1)
    negatives = logits[~labels.bool()].view(bsz * 2, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label = torch.zeros(bsz * 2, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, label)

    return loss


def supcon_pos_neg_loss(z1, z2, z3, z4, temperature=0.5):
    # z3 is pos, z4 is neg
    bsz = z1.shape[0]
    # bsz2 = bsz * 2
    bsz3 = bsz * 3

    labels = torch.cat([torch.arange(bsz) for i in range(3)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    z_anchor = torch.cat([z1, z2, z3], dim=0)
    z_anchor = F.normalize(z_anchor, dim=1)

    z_contra = torch.cat([z1, z2, z3, z4], dim=0)
    z_contra = F.normalize(z_contra, dim=1)

    logits = z_anchor @ z_contra.T
    mask = torch.cat([torch.eye(bsz3, dtype=torch.bool), torch.zeros(bsz3, bsz, dtype=torch.bool)], dim=1).cuda()
    labels = torch.cat([labels, torch.zeros(bsz3, bsz)], dim=1).cuda()
    labels = labels[~mask].view(bsz3, -1)
    logits = logits[~mask].view(bsz3, -1)
    positives = logits[labels.bool()].view(bsz3, -1)
    negatives = logits[~labels.bool()].view(bsz3, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label0 = torch.zeros(bsz3, dtype=torch.long).cuda()
    loss0 = torch.nn.functional.cross_entropy(logits, label0)
    label1 = torch.ones(bsz3, dtype=torch.long).cuda()
    loss1 = torch.nn.functional.cross_entropy(logits, label1)
    loss = (loss0 + loss1) / 2

    return loss


def simclr_loss(z1, z2, temperature=0.5, base_temperature=0.5):
    labels = torch.cat([torch.arange(z1.shape[0]) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()
    
    z = torch.cat([z1, z2], dim=0)
    z = torch.nn.functional.normalize(z, dim=1)

    logits = z @ z.T
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    logits = logits[~mask].view(logits.shape[0], -1)
    positives = logits[labels.bool()].view(labels.shape[0], -1)
    negatives = logits[~labels.bool()].view(labels.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, label)
    loss = temperature / base_temperature * loss

    return loss


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        if encoder:
            self.encoder = encoder
        else:
            if args.backbone == 'resnet18':
                from resnet import resnet18
                self.encoder = resnet18()
                feature_dim = 512
            elif args.backbone == 'resnet50':
                from resnet import resnet50
                self.encoder = resnet50()
                feature_dim = 512 * 4
            else:
                raise ValueError('Unknown backbone: {}'.format(args.backbone))
        if args.use_supcon_projector:
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, dim_proj[1]),
            )
        else:
            self.projector = ProjectionMLP(feature_dim, dim_proj[0], dim_proj[1])
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )
        if args.loss == 'simclr':
            self.predictor2 = nn.Sequential(nn.Linear(feature_dim, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 4))  # output layer
        else:
            self.predictor2 = nn.Sequential(nn.Linear(feature_dim, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, 4))  # output layer

    def forward(self, x):
        return self.net(x)


def get_viewmaker(args, which_viewmaker='none', branch=None, eval_mode=False):
    if which_viewmaker == 'gan':  # GAN on-the-fly
        from view_generator import StyleGANGenerator, get_gan_models
        if args.pretrained_encoder_path:
            viewmaker_encoder = Branch(args, encoder=None).cuda()
            saved_dict = torch.load(os.path.join(args.pretrained_encoder_path))['state_dict']
            viewmaker_encoder.load_state_dict(saved_dict)
            freeze_encoder = True
        else:
            viewmaker_encoder = branch
            freeze_encoder = eval_mode
        gan_gen, gan_enc = get_gan_models('cuda')
        viewmaker = StyleGANGenerator(
            gan_generator=gan_gen,
            gan_encoder=gan_enc,
            simclr_encoder=viewmaker_encoder,
            idinvert_steps=args.idinvert_steps,
            boundary_steps=args.boundary_steps,
            boundary_epsilon=args.boundary_epsilon,
            fgsm_stepsize=args.fgsm_stepsize,
            freeze_encoder=freeze_encoder,
        )
        viewmaker_kwargs = {}
    elif which_viewmaker == 'gaussian':
        from view_generator import GaussianViewGenerator, get_gan_models
        gan_gen, _ = get_gan_models('cuda', no_encoder=True)
        viewmaker = GaussianViewGenerator(
            gan_generator=gan_gen,
            gan_encoder=None,
            simclr_encoder=None,
            idinvert_steps=0,
            boundary_steps=0,
            boundary_epsilon=0,
            fgsm_stepsize=0,
            freeze_encoder=True,
        )
        viewmaker_kwargs = {
            'noise_std': args.gaussian_noise_std,
        }
    elif which_viewmaker == 'stn':
        from view_generator import SpatialTransformerGenerator
        if args.pretrained_encoder_path:
            viewmaker_encoder = Branch(args, encoder=None).cuda()
            saved_dict = torch.load(os.path.join(args.pretrained_encoder_path))['state_dict']
            viewmaker_encoder.load_state_dict(saved_dict)
            freeze_encoder = True
        else:
            viewmaker_encoder = branch
            freeze_encoder = eval_mode
        viewmaker = SpatialTransformerGenerator(
            simclr_encoder=viewmaker_encoder,
            boundary_epsilon=args.boundary_epsilon,
            fgsm_stepsize=args.fgsm_stepsize,
            freeze_encoder=freeze_encoder,
        )
        viewmaker_kwargs = {
            'init_scale': args.stn_init_scale,
        }
    elif which_viewmaker == 'stnrand':
        from view_generator import RandomSpatialTransformerGenerator
        viewmaker = RandomSpatialTransformerGenerator()
        viewmaker_kwargs = {
            'init_scale': args.stn_init_scale,
            'noise_std': args.stn_noise_std,
        }
    elif which_viewmaker == 'stnrand2':
        from view_generator import RandomSpatialTransformerGenerator2
        viewmaker = RandomSpatialTransformerGenerator2()
        viewmaker_kwargs = {
            'scale': args.stn_area_scale,
            'ratio': args.stn_aspect_ratio,
            'noise_std': args.stn_noise_std,
            'clamp': args.stn_area_scale_clamp,
            'strict': args.stn_area_scale_strict,
            'same_along_batch': args.stn_same_along_batch,
        }
    elif which_viewmaker == 'stncrop':
        from view_generator import RandomResizedCropGenerator
        viewmaker = RandomResizedCropGenerator()
        viewmaker_kwargs = {
            'scale': args.stn_area_scale,
            'ratio': args.stn_aspect_ratio,
        }
    else:
        viewmaker = lambda x, w, **kwargs: x
        viewmaker_kwargs = {}
    return viewmaker, viewmaker_kwargs


def knn_loop(encoder, train_loader, test_loader):
    accuracy = knn_monitor(
        net=encoder.cuda(),
        memory_data_loader=train_loader,
        test_data_loader=test_loader,
        device='cuda',
        k=200,
        hide_progress=True)
    return accuracy


def ssl_loop(args, encoder=None, logger=None):
    if args.checkpoint_path:
        print('checkpoint provided => moving to evaluation')
        main_branch = Branch(args, encoder=encoder).cuda()
        state_dict_key = 'state_dict' if args.load_old_checkpoint else 'model_state_dict'
        saved_dict = torch.load(os.path.join(args.checkpoint_path))[state_dict_key]
        main_branch.load_state_dict(saved_dict)
        if args.linear_probe:
            ep = Path(args.checkpoint_path).stem
            filename_suffix = f'_ep={ep}_lr={args.lr_probe}_mom={args.mom_probe}_wd={args.wd_probe}'
            file_to_update = logging_file(os.path.join(args.path_dir, f'{args.eval_log_prefix}eval{filename_suffix}.log.txt'), 'a+')
            file_to_update.write(f'evaluating {args.checkpoint_path}\n')
            return main_branch, file_to_update
        else:
            assert args.finetune

    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = logging_file(os.path.join(args.path_dir, 'train_and_eval.log.txt'), 'a+')

    # dataset
    train_dataset, post_transform3 = get_train_dataset_and_transforms(args)
    print(train_dataset.transform0)
    print(train_dataset.transform1)
    print(train_dataset.transform3, post_transform3)
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(train_dataset))[:int(args.limit_train_batches * len(train_dataset))]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"limit train dataset to len={len(train_dataset)} (random subset, seed={args.seed}, random={random.random():.4f})")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_default_dataset(args.dataset, True, None),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_default_dataset(args.dataset, False, None),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # models

    main_branch = Branch(args, encoder=encoder).cuda()
    predictor, pred_optimizer = None, None

    viewmaker, viewmaker_kwargs = get_viewmaker(args, args.which_view_generator, main_branch)

    if args.loss == 'simsiam':
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        predictor = PredictionMLP(dim_proj[1], args.dim_pred, dim_proj[1]).cuda()

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    if args.loss == 'simsiam':
        pred_optimizer = torch.optim.SGD(
            predictor.parameters(),
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
            weight_decay=args.wd
        )

    # macros
    backbone = main_branch.encoder
    projector = main_branch.projector

    # ================ resume ================
    start_epoch = 1
    time_offset = 0
    if args.resume:
        ckpt_path = get_last_checkpoint(
            ckpt_dir=os.path.join(args.log_dir, 'weights'),
            ckpt_ext='.pt',
            latest='latest.pt' if args.save_latest else None,
        )
        print(f"resuming from checkpoint {ckpt_path}...")
        assert os.path.exists(ckpt_path), f'{ckpt_path} does not exist'
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1 if args.start_epoch is None else args.start_epoch
        time_offset = ckpt['time_elapsed']
        main_branch.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if args.loss == 'simsiam':
            predictor.load_state_dict(ckpt['predictor_state_dict'])
            pred_optimizer.load_state_dict(ckpt['pred_optimizer_state_dict'])
        print(f"starting from epoch {start_epoch}...")

    # logging
    start_time = time.time()
    if not args.resume:
        os.makedirs(args.path_dir, exist_ok=True)
        torch.save({
            'epoch': 0,
            'time_elapsed': time.time() - start_time + time_offset,
            'model_state_dict': main_branch.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_state_dict': predictor.state_dict() if args.loss == 'simsiam' else None,
            'pred_optimizer_state_dict': pred_optimizer.state_dict() if args.loss == 'simsiam' else None,
        }, os.path.join(args.path_dir, 'weights', '0.pt'))
    scaler = GradScaler()

    criterion = None
    if args.which_loss == 'simclr_supcon':
        from losses import SupConLoss
        criterion = SupConLoss(temperature=args.temp, base_temperature=args.base_temp)

    # training
    for e in range(start_epoch, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.loss == 'simsiam':
            predictor.train()

        # epoch
        for it, (inputs, w, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # synthesize views
            if args.use_view_generator:
                w = w.cuda()
                for j in range(args.n_views_gan):  # TODO: parallelize by concat? -> no, because of BN
                    view_j = viewmaker(inputs[3 + j].cuda(), w, **viewmaker_kwargs)
                    inputs[3 + j] = post_transform3(view_j) if post_transform3 is not None else view_j
            if args.replace_expert_views and 0 in args.replace_expert_views:
                inputs[0] = inputs.pop(3)
            if args.replace_expert_views and 1 in args.replace_expert_views:
                inputs[1] = inputs.pop(3)

            # adjust
            lr = adjust_learning_rate(
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                base_lr=args.lr * args.bsz / 256,
                optimizer=optimizer,
                loader=train_loader,
                step=it)
            # zero grad
            main_branch.zero_grad()
            if args.loss == 'simsiam':
                predictor.zero_grad()

            def forward_step():
                x1 = inputs[0].cuda()
                x2 = inputs[1].cuda()
                if not args.concat_inputs:  # concat will done in the loss
                    b1 = backbone(x1)
                    b2 = backbone(x2)
                    z1 = projector(b1)
                    z2 = projector(b2)

                # forward pass
                if args.which_loss == 'simclr':
                    if args.concat_inputs:
                        x = torch.cat([x1, x2], dim=0)
                        b = backbone(x)
                        z = projector(b)
                        z = F.normalize(z, dim=1)
                        z1, z2 = torch.split(z, args.bsz, dim=0)
                    loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                elif args.which_loss == 'simclr_all':
                    if args.concat_inputs:
                        x = torch.cat([x1, x2], dim=0)
                        b = backbone(x)
                        z = projector(b)
                        z = F.normalize(z, dim=1)
                        z1, z2 = torch.split(z, args.bsz, dim=0)
                    loss = simclr_loss(z1, z2, args.temp, args.base_temp)
                elif args.which_loss == 'simclr_supcon':
                    x = torch.cat([x1, x2], dim=0)
                    b = backbone(x)
                    z = projector(b)
                    z = F.normalize(z, dim=1)
                    f1, f2 = torch.split(z, args.bsz, dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss = criterion(features)
                elif args.which_loss == 'simclr+pos':
                    loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                    z1 = F.normalize(z1, dim=1)
                    for j in range(args.n_views_gan):
                        xj = inputs[3 + j]
                        bj = backbone(xj)
                        zj = projector(bj)
                        zj = F.normalize(zj, dim=1)
                        loss -= torch.mean(torch.sum(z1 * zj, dim=1)) / args.n_views_gan
                elif args.which_loss == 'simclr-neg':
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)
                    x3 = inputs[3]
                    b3 = backbone(x3)
                    z3 = projector(b3)
                    z3 = F.normalize(z3, dim=1)
                    loss = supcon_neg_loss(z1, z2, z3)
                elif args.which_loss == 'simclr+pos-neg':
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)
                    x3 = inputs[3]
                    b3 = backbone(x3)
                    z3 = projector(b3)
                    z3 = F.normalize(z3, dim=1)
                    x4 = inputs[4]
                    b4 = backbone(x4)
                    z4 = projector(b4)
                    z4 = F.normalize(z4, dim=1)
                    loss = supcon_pos_neg_loss(z1, z2, z3, z4)
                elif args.which_loss == 'simclr+supcon':
                    x3 = inputs[3]
                    if args.concat_inputs:
                        x = torch.cat([x1, x2, x3], dim=0)
                        b = backbone(x)
                        z = projector(b)
                        z = F.normalize(z, dim=1)
                        z1, z2, z3 = torch.split(z, args.bsz, dim=0)
                    else:
                        b3 = backbone(x3)
                        z3 = projector(b3)
                    # z1 = F.normalize(z1, dim=1)
                    # z2 = F.normalize(z2, dim=1)
                    # z3 = F.normalize(z3, dim=1)
                    loss = supcon_loss(z1, z2, z3)
                elif args.which_loss == 'simclr_3_views':
                    x3 = inputs[2].cuda()  # NOTE: reuse inputs[2]
                    if args.concat_inputs:
                        x = torch.cat([x1, x2, x3], dim=0)
                        b = backbone(x)
                        z = projector(b)
                        z = F.normalize(z, dim=1)
                        z1, z2, z3 = torch.split(z, args.bsz, dim=0)
                    else:
                        b3 = backbone(x3)
                        z3 = projector(b3)
                    loss = supcon_loss(z1, z2, z3)
                elif args.which_loss == 'simclr-rep_vg4+a2-diag':
                    x3 = inputs[2].cuda()  # NOTE: reuse inputs[2]
                    if args.concat_inputs:
                        x = torch.cat([x1, x2, x3], dim=0)
                        b = backbone(x)
                        z = projector(b)
                        z = F.normalize(z, dim=1)
                        z1, z2, z3 = torch.split(z, args.bsz, dim=0)
                    else:
                        b3 = backbone(x3)
                        z3 = projector(b3)
                        z1 = F.normalize(z1, dim=1)
                        z2 = F.normalize(z2, dim=1)
                        z3 = F.normalize(z3, dim=1)
                    loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                    inner_prod = torch.mean(torch.sum(z1 * z3, dim=1)) / 2 + torch.mean(torch.sum(z2 * z3, dim=1)) / 2
                    loss = loss - inner_prod
                elif args.which_loss == 'simsiam':
                    p1 = predictor(z1)
                    p2 = predictor(z2)
                    loss = negative_cosine_similarity_loss(p1, z2) / 2 + negative_cosine_similarity_loss(p2, z1) / 2
                else:
                    raise

                if args.lmbd > 0:
                    rotated_images, rotated_labels = rotate_images(inputs[2])
                    b = backbone(rotated_images)
                    logits = main_branch.predictor2(b)
                    rot_loss = F.cross_entropy(logits, rotated_labels)
                    loss += args.lmbd * rot_loss
                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.loss == 'simsiam':
                    scaler.step(pred_optimizer)

            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()
                if args.loss == 'simsiam':
                    pred_optimizer.step()

        if args.fp16:
            with autocast():
                knn_acc = knn_loop(backbone, memory_loader, test_loader)
        else:
            knn_acc = knn_loop(backbone, memory_loader, test_loader)

        line_to_print = (
            f'epoch: {e} | knn_acc: {knn_acc:.3f} | '
            f'loss: {loss.item():.3f} | lr: {lr:.6f} | '
            f'time_elapsed: {(time.time() - start_time + time_offset) / 3600:.3f}h'
        )
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
        print(line_to_print)
        if logger is not None:
            logger.log(
                {
                    'epoch': e,
                    'knn_acc': knn_acc,
                    'loss': loss.item(),
                    'lr': lr,
                    'time_elapsed': (time.time() - start_time + time_offset) / 3600,  # in hours
                },
                step=e + args.wandb_step_offset,  # NOTE: syncs with epoch number, 
            )  # this also prevents duplicated logging when resuming from a previous checkpoint
        if args.save_latest:  # save every epoch
            torch.save({
                'epoch': e,
                'time_elapsed': time.time() - start_time + time_offset,
                'model_state_dict': main_branch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'predictor_state_dict': predictor.state_dict() if args.loss == 'simsiam' else None,
                'pred_optimizer_state_dict': pred_optimizer.state_dict() if args.loss == 'simsiam' else None,
            }, os.path.join(args.path_dir, 'weights', f'latest.pt'))
        if e % args.save_every == 0:
            torch.save({
                'epoch': e,
                'time_elapsed': time.time() - start_time + time_offset,
                'model_state_dict': main_branch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'predictor_state_dict': predictor.state_dict() if args.loss == 'simsiam' else None,
                'pred_optimizer_state_dict': pred_optimizer.state_dict() if args.loss == 'simsiam' else None,
            }, os.path.join(args.path_dir, 'weights', f'{e}.pt'))

    return main_branch, file_to_update


def eval_loop(args, branch, file_to_update, ind=None, logger=None):
    args.n_views_gan = min(1, args.n_views_gan)
    encoder = copy.deepcopy(branch.encoder)
    toggle_grad(encoder, False)
    if args.eval_train_transform:
        viewmaker, viewmaker_kwargs = get_viewmaker(args, args.which_view_generator, branch, eval_mode=True)
    else:
        viewmaker = None
        viewmaker_kwargs = None

    # dataset
    normalize, _, _ = get_default_transform(args.dataset)
    train_transform0, train_transform1, train_transform3, post_transform3 = None, None, None, None
    if args.eval_train_transform:
        train_transform = None
        train_transform0, _ = get_transforms(args, args.eval_train_transform0)
        train_transform1, _ = get_transforms(args, args.eval_train_transform1)
        train_transform3, post_transform3 = get_transforms(args, args.eval_train_transform3)
    else:
        train_transform = T.Compose([
            T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    if args.eval_test_use_original:
        test_transform = T.Compose([
            T.ToTensor(),
            normalize
        ])
    else:
        test_transform = T.Compose([
            T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(32),
            T.ToTensor(),
            normalize
        ])
    if args.eval_train_transform:
        from utils_data import MultiViewDataset
        train_dataset = MultiViewDataset(
            args.data_path,
            args.latent_path,
            args.view_paths,
            transform0=train_transform0,
            transform1=train_transform1,
            transform2=None,
            transform3=train_transform3,
            n_views=args.n_views_gan,
            train=True,
            sample_from_mixed=args.sample_from_mixed,
            sample_original=args.sample_original,
        )
        transform_probabilities = np.array([float(p) for p in args.eval_train_transform_probabilities.split(',')])
        transform_probabilities /= transform_probabilities.sum()
        print(f"linear probe: sampling view_{{0,1,3}} with probabilities: {transform_probabilities}")
    else:
        # train_dataset = torchvision.datasets.CIFAR10('../data', train=True, transform=train_transform, download=True)
        train_dataset = get_default_dataset(args.dataset, train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.eval_train_bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        # dataset=torchvision.datasets.CIFAR10('../data', train=False, transform=test_transform, download=True),
        dataset=get_default_dataset(args.dataset, False, test_transform),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers
    )

    if args.dataset == 'cifar10':
        classifier = nn.Linear(512, 10).cuda()
    elif args.dataset == 'cifar100':
        classifier = nn.Linear(512, 100).cuda()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=args.mom_probe,
        lr=args.lr_probe,
        weight_decay=args.wd_probe,
    )
    scaler = GradScaler()

    # training
    start_time = time.time()

    for e in range(1, 101):

        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, batch in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            if args.eval_train_transform:
                inputs, w, y = batch
                inputs = [x.cuda() for x in inputs]
                if args.quick_test:
                    if post_transform3 is not None:
                        inputs[3] = post_transform3(inputs[3])
                    k = np.random.choice([0, 1, 3], p=transform_probabilities)
                    images = inputs[k]
                else:
                    w = w.cuda()
                    for j in range(args.n_views_gan):
                        view_j = viewmaker(inputs[3 + j], w, **viewmaker_kwargs)
                        inputs[3 + j] = post_transform3(view_j) if post_transform3 is not None else view_j
                    index = np.random.choice([0, 1, 3], inputs[0].size(0), p=transform_probabilities)
                    images = torch.stack([inputs[index[b]][b] for b in range(inputs[0].size(0))])
            else:
                images, y = batch
                images = images.cuda()
            # adjust
            adjust_learning_rate(
                epochs=100,
                warmup_epochs=0,
                base_lr=args.lr_probe,
                optimizer=optimizer,
                loader=train_loader,
                step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(images)
                logits = classifier(b)
                loss = F.cross_entropy(logits, y.cuda())
                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            b = encoder(images.cuda())
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.cuda())
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.cuda()).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f} | '
                f'time_elapsed: {(time.time() - start_time):.1f}s'
            )
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    return accuracy  # return acc for the last epoch (100)


def main(args, logger=None):
    fix_seed(args.seed)
    branch, file_to_update = ssl_loop(args, logger=logger)
    accs = []
    seeds = np.arange(args.eval_loops)
    for i in seeds:
        accs.append(eval_loop(args, copy.deepcopy(branch), file_to_update, i))
    line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)
    if not NO_LOGGING and logger is not None:
        import wandb
        table = wandb.Table(data=[accs + [np.mean(accs)]], columns=[f'seed={i}' for i in seeds] + ['mean'])
        logger.log({'linear_probe': table})
        # table_bar = [[f'{i}', acc] for i, acc in zip(seeds, accs)] + [['mean', np.mean(accs)]]
        # table_bar = wandb.Table(data=table_bar, columns=['seed', 'accuracy'])
        # logger.log({'linear_probe_bar': wandb.plot.bar(table_bar, 'seed', 'accuracy', title='linear probe top-1 accuracy')})
        logger.finish()


if __name__ == '__main__':
    """
    inputs: [x0, x1, x2, x3, ...]
    x0, x1 are two expert views
    x2 are rotation views reserved for essl
    x3 and after are for (cached) GAN generated views

    To replace expert views with GAN generated views:
    --which_loss='simclr'
    --replace_expert_views
    --which_transform0=''
    --which_transform3='gan+horizontalflip'

    cached views:
    --which_view_generator='cache'

    on-the-fly:
    --which_view_generator='gan' --n_views_gan=1 --sample_from_mixed

    To append GAN generated views to expert views:
    --which_loss='simclr+pos'
    --which_transform3='gan+horizontalflip'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,2048', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--save_every', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--path_dir', default='logs/simclr_baseline', type=str, help='log dir')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.0, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', default=None, type=str, help='cuda device ids to use')
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--eval_loops', default=5, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_epoch', default=None, type=int, help='resume from this epoch')
    parser.add_argument('--save_latest', action='store_true')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='simclr-cifar10', type=str)
    parser.add_argument('--wandb_step_offset', default=-1, type=int,
        help='this is to be compatible with old runs, since epoch starts from 1.')
    parser.add_argument('--use_view_generator', action='store_true')
    parser.add_argument('--which_view_generator', default='none', type=str, choices=['none', 'gan', 'gaussian', 'cache', 'stn', 'stnrand', 'stnrand2', 'stncrop'])
    parser.add_argument('--data_path', type=str, default='data/c10_data2.pkl')
    parser.add_argument('--latent_path', type=str, default='data/c10_latent.pkl')
    parser.add_argument('--view_paths', type=str, default='')
    parser.add_argument('--sample_from_mixed', action='store_true')
    parser.add_argument('--pretrained_encoder_path', default=None, type=str)
    parser.add_argument('--fgsm_stepsize', default=0.1, type=float)
    parser.add_argument('--idinvert_steps', default=0, type=int)
    parser.add_argument('--boundary_steps', default=1, type=int)
    parser.add_argument('--boundary_epsilon', default=0.5, type=float)
    parser.add_argument('--n_views_gan', default=0, type=int, help='number of generated views, view_id >= 3')
    parser.add_argument('--which_transform0', type=str, default='resizedcrop+horizontalflip+colorjitter+grayscale')
    parser.add_argument('--which_transform1', type=str, default='resizedcrop+horizontalflip+colorjitter+grayscale')
    parser.add_argument('--which_transform2', type=str, default='resizedcrophalf+horizontalflip+colorjitter+grayscale')
    parser.add_argument('--which_transform3', type=str, default='horizontalflip+gan')
    parser.add_argument('--which_loss', type=str, default='simclr',
        choices=['simclr', 'simsiam', 'simclr+pos', 'simclr+supcon',
            'simclr_all', 'simclr_supcon', 'simclr-neg', 'simclr+pos-neg',
            'simclr_3_views', 'simclr-rep_vg4+a2-diag'],
        help='--loss is kept for compatibility with the original code')
    parser.add_argument('--code_to_save', type=str, default='utils_data.py,view_generator.py',
        help='source files to save, note that main2.py will be saved in print_args')
    parser.add_argument('--linear_probe', action='store_true', help='run linear probing')
    parser.add_argument('--lr_probe', default=30, type=float)
    parser.add_argument('--wd_probe', default=0, type=float)
    parser.add_argument('--mom_probe', default=0.9, type=float)
    parser.add_argument('--load_old_checkpoint', action='store_true')
    parser.add_argument('--eval_test_use_original', action='store_true')
    parser.add_argument('--eval_train_transform', action='store_true')
    parser.add_argument('--eval_train_transform_probabilities', type=str, default='0,1,0', help="probabilities for sampling view_{0,1,3}")
    parser.add_argument('--eval_train_transform0', type=str, default='horizontalflip')
    parser.add_argument('--eval_train_transform1', type=str, default='resizedcrop+horizontalflip')
    parser.add_argument('--eval_train_transform3', type=str, default='horizontalflip+gan')
    parser.add_argument('--eval_train_bsz', default=256, type=int)
    parser.add_argument('--eval_log_prefix', type=str, default='')

    parser.add_argument('--stn_init_scale', default=1., type=float)
    parser.add_argument('--stn_noise_std', default=0.1, type=float)
    parser.add_argument('--stn_area_scale', type=str, default='0.2,1')
    parser.add_argument('--stn_area_scale_clamp', action='store_true')
    parser.add_argument('--stn_area_scale_strict', action='store_true')
    parser.add_argument('--stn_aspect_ratio', type=str, default='0.75,1.333')
    parser.add_argument('--stn_same_along_batch', action='store_true')
    parser.add_argument('--replace_expert_views', type=str, default=None, help="if '0,1', replace 0 and 1 with 3")
    parser.add_argument('--sample_original', action='store_true', help='allow sampling views with replacement, [0] * n_views')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--temp', default=0.5, type=float)
    parser.add_argument('--base_temp', default=0.5, type=float)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--use_supcon_projector', action='store_true')
    parser.add_argument('--use_pos_neg_view', action='store_true')
    parser.add_argument('--pos_view_paths', type=str, default='')
    parser.add_argument('--neg_view_paths', type=str, default='')
    parser.add_argument('--add_smallcrop', action='store_true')
    parser.add_argument('--concat_inputs', action='store_true')
    parser.add_argument('--gaussian_noise_std', default=0.2, type=float)
    parser.add_argument('--n_cache_gan', default=8, type=int)
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()

    args.log_dir = Path(args.path_dir)  # NOTE: for compatibility
    args.use_wandb = not args.no_wandb
    # args.save_latest = True  # TODO: hardcoded

    # ============ modify args =============
    if args.debug:
        args.num_workers = 0
        args.eval_loops = 1
        args.limit_train_batches = 0.1
    
    if args.which_loss == 'simclr_supcon':
        args.concat_inputs = True
    
    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    for i in range(4):
        transform_i = getattr(args, f'which_transform{i}')
        print(f"which_transform{i} = {transform_i.split('+')}")
    args.view_paths = [s for s in args.view_paths.split(',') if s != '']
    print(f"view_paths = {args.view_paths}")
    if args.use_pos_neg_view:
        args.pos_view_paths = [s for s in args.pos_view_paths.split(',') if s != '']
        print(f"pos_view_paths = {args.pos_view_paths}")
        args.neg_view_paths = [s for s in args.neg_view_paths.split(',') if s != '']
        print(f"neg_view_paths = {args.neg_view_paths}")
    if args.replace_expert_views:
        args.replace_expert_views = [int(s) for s in args.replace_expert_views.split(',') if s != '']
        print(f"expert views {args.replace_expert_views} will be replaced by view_{{{','.join([str(i+3) for i in range(args.n_views_gan)])}}}...")
    args.stn_area_scale = [float(s) for s in args.stn_area_scale.split(',') if s != '']
    args.stn_aspect_ratio = [float(s) for s in args.stn_aspect_ratio.split(',') if s != '']
    if args.use_view_generator and args.which_view_generator == 'stnrand2':
        print(f"stnrand2 will be used with scale={args.stn_area_scale} and ratio={args.stn_aspect_ratio}.")
    if not args.use_view_generator and args.which_view_generator != 'none':
        print(f"view generator {args.which_view_generator} will be ignored, set to none.")
        args.which_view_generator = 'none'

    if args.linear_probe:
        NO_LOGGING = True
        args.use_wandb = False
        if args.checkpoint_path == 'self':
            args.checkpoint_path = args.log_dir / 'weights' / '800.pt'
        assert args.checkpoint_path and os.path.exists(args.checkpoint_path)

    # ========== check args ==========
    if args.use_view_generator and args.which_view_generator == 'gan':
        assert args.n_views_gan == 1 and args.sample_original
        assert args.which_transform3.startswith('gan'), 'should be no other transforms before gan, since view_generator uses paired latents'

    # get_default_transform(args.dataset)

    # ========== setup logger ==========
    if not NO_LOGGING:
        os.makedirs(os.path.join(args.log_dir), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'weights'), exist_ok=True)
        logger = setup_wandb(args) if args.use_wandb else None
    else:
        logger = None

    # ========== backup args and code ==========
    if not NO_LOGGING:
        print_args(parser, args)
        for filename in args.code_to_save.split(','):
            shutil.copy(filename, os.path.join(args.log_dir, 'src', filename+'.txt'))
    if logger is not None:
        logger.log_code(".")

    main(args, logger)
