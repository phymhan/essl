from pathlib import Path
import argparse
import os
import sys
import shutil
import random
import subprocess
import time
import json
import math
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

from utils import gather_from_all, GaussianBlur, Solarization
from utils import print_args, setup_wandb, fix_seed

import pdb
st = pdb.set_trace

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def data_sampler(dataset, shuffle, distributed, drop_last=True):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle, drop_last=drop_last)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def main_worker(gpu, args):
    DISTRIBUTED = args.distributed
    if DISTRIBUTED:
        args.rank += gpu
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    else:
        global gather_from_all
        gather_from_all = lambda x: x

    args.checkpoint_dir = args.checkpoint_dir
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

        # NOTE: wandb has to be initialized after spawning processes
        logger = setup_wandb(args) if args.use_wandb else None
        if logger is not None:
            logger.log_code(".")
        print(f"gpu: {gpu}, logging to {args.log_dir}, logger={logger}")

    if args.rank != 0 and DISTRIBUTED:
        logger = None

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = SimCLR(args).cuda(gpu)
    if DISTRIBUTED:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_module = model.module
    else:
        model_module = model

    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                    weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm)
    
    viewmaker, viewmaker_kwargs = get_viewmaker(args, args.which_view_generator)

    time_offset = 0
    start_epoch = 0
    global_step = 0
    if args.resume:
        from utils import get_last_checkpoint
        ckpt_path = get_last_checkpoint(
            ckpt_dir=os.path.join(args.log_dir, 'weights'),
            ckpt_ext='.pth',
            latest=None,
        ) if args.checkpoint_path is None else args.checkpoint_path
        print(f"resuming from checkpoint {ckpt_path}...")
        assert os.path.exists(ckpt_path), f'{ckpt_path} does not exist'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step'] + 1
        time_offset = ckpt['time_elapsed']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"starting from epoch {start_epoch}...")

    dataset, post_transform3 = get_train_dataset_and_transforms(args)
    if args.rank == 0:
        print(dataset)
        print(f"post_transform3: {post_transform3.transform}")
    # TODO: check if this is correct
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(dataset))[:int(args.limit_train_batches * len(dataset))]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"limit train dataset to len={len(dataset)} (random subset, seed={args.seed}, random={random.random():.4f})")
    sampler = data_sampler(dataset, True, DISTRIBUTED, True)
    
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        if DISTRIBUTED:
            sampler.set_epoch(epoch)

        for step, (inputs, latents, labels) in enumerate(loader, start=epoch * len(loader)):
            y1, y2, y3 = inputs[0], inputs[1], inputs[2]
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            # synthesize views
            if args.use_view_generator:
                # latents = latents.cuda(gpu, non_blocking=True)
                for j in range(args.n_views_gan):
                    view_j = viewmaker(inputs[3 + j].cuda(gpu, non_blocking=True), latents, **viewmaker_kwargs)
                    inputs[3 + j] = post_transform3(view_j) if post_transform3 is not None else view_j
            if args.replace_expert_views and 0 in args.replace_expert_views:
                inputs[0] = inputs.pop(3)
            if args.replace_expert_views and 1 in args.replace_expert_views:
                inputs[1] = inputs.pop(3)

            if args.rotation:
                y3 = y3.cuda(gpu, non_blocking=True)
                rotated_images, rotated_labels = rotate_images(y3, gpu)

            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, acc, z1, z2 = model.forward(y1, y2, labels)

                if args.rotation:
                    logits = model_module.forward_rotation(rotated_images)
                    rot_loss = torch.nn.functional.cross_entropy(logits, rotated_labels)
                    loss += args.rotation * rot_loss
                
                if args.which_loss == 'simclr+pos':
                    # TODO: view 0 or 1 ? use 1 for now (z2)
                    za = torch.nn.functional.normalize(z2, dim=1)
                    for j in range(args.n_views_gan):
                        xb = inputs[3 + j]
                        rb = model_module.backbone(xb)
                        zb = model_module.projector(rb)
                        zb = torch.nn.functional.normalize(zb, dim=1)
                        loss -= torch.mean(torch.sum(za * zb, dim=1)) / args.n_views_gan

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                if DISTRIBUTED:
                    torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}')
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)
                    if logger is not None:
                        stats['time'] = stats['time'] / 3600  # convert to hours
                        # stats.pop('step')
                        logger.log(
                            stats,
                            step=global_step,
                        )

            global_step += 1  # global step is incremented after logging

        if args.rank == 0 and epoch % args.save_every == 0:
            # save checkpoint
            state = dict(
                epoch=epoch,
                global_step=global_step,
                time_elapsed=time.time() - start_time + time_offset,
                model=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
            torch.save(state, os.path.join(args.log_dir, 'weights', f'checkpoint_{epoch}.pth'))

    if args.rank == 0:
        # save final model
        torch.save(dict(backbone=model_module.backbone.state_dict(),
                        projector=model_module.projector.state_dict(),
                        head=model_module.online_head.state_dict()),
                args.checkpoint_dir / 'resnet50_last.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)  # 10 epochs
    base_lr = args.learning_rate #* args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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

        return loss, acc, z1, z2

    def forward_rotation(self, x):
        b = self.backbone(x)
        logits = self.rotation_projector(b)

        return logits


def infoNCE(nn, p, temperature=0.2):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class Transform:
    def __init__(self, size=224, scale=[0.05, 0.14], rotation=0):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.rotation = rotation
        self.transform_rotation = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(scale[0], scale[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            normalize,
        ]) if rotation else None

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_rotation(x) if self.rotation else 0
        return y1, y2, y3


# helper functions
def get_transform_list(which_transform, size=224):
    if isinstance(which_transform, str):
        which_transform = which_transform.replace(',', '+').split('+')
    transform_list = []
    for t in which_transform:
        t = t.lower()
        if t == 'centercrop':  # TODO: what should we use?
            transform_list += [
                transforms.Resize(size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(size),
            ]
        elif t == 'randomcrop':
            transform_list += [
                transforms.Resize(size, interpolation=Image.BICUBIC),
                transforms.RandomCrop(size),
            ]
        elif t == 'resizedcrop':
            transform_list.append(transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC))
        elif t == 'resizedcrophalf':  # NOTE: hardcoded size and scale
            transform_list.append(transforms.RandomResizedCrop(96, scale=(0.05, 0.14)))
        elif t == 'horizontalflip':
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        elif t == 'colorjitter':
            transform_list.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)], p=0.8))
        elif t == 'grayscale':
            transform_list.append(transforms.RandomGrayscale(p=0.2))
        elif t.startswith('gaussianblur'):
            p = float(t.split('=')[1]) if '=' in t else 0.1
            if p > 0:
                transform_list.append(GaussianBlur(p=p))
        elif t.startswith('solarization'):
            p = float(t.split('=')[1]) if '=' in t else 0.0
            if p > 0:
                transform_list.append(Solarization(p=p))
        elif t.startswith('totensor'):
            transform_list.append(transforms.ToTensor())
        elif t.startswith('normalize'):
            transform_list.append(normalize)
    return transform_list


class BatchwiseTransform:
    def __init__(self, transform):
        # perform random transform along batch dimension
        self.transform = transform

    def __call__(self, x):
        # x: [B, C, H, W]
        y = [self.transform(i) for i in x]
        return torch.stack(y, dim=0)


def get_transforms(which_transform='', size=224):
    if isinstance(which_transform, str):
        which_transform = which_transform.replace(',', '+').split('+')
    if 'gan' in which_transform:
        index = which_transform.index('gan')
        pre_transform = get_transform_list(which_transform[:index], size=size) + \
            [transforms.ToTensor()]
        post_transform = get_transform_list(which_transform[index + 1:], size=size) + \
            [normalize]
    else:
        pre_transform = get_transform_list(which_transform, size=size) + \
            [transforms.ToTensor(), normalize]
        post_transform = None
    if pre_transform is not None:
        pre_transform = transforms.Compose(pre_transform)
    if post_transform is not None:
        post_transform = BatchwiseTransform(transforms.Compose(post_transform))
    return pre_transform, post_transform


def get_train_dataset_and_transforms(args):
    # transforms for view_0
    pre_transform0, post_transform0 = get_transforms(args.which_transform0, args.image_size)
    assert post_transform0 is None, 'post_transform0 should be None'

    # transforms for view_1
    pre_transform1, post_transform1 = get_transforms(args.which_transform1, args.image_size)
    assert post_transform1 is None, 'post_transform1 should be None'

    # transforms for view_2
    pre_transform2, post_transform2 = get_transforms(args.which_transform2, args.image_size)
    assert post_transform2 is None, 'post_transform2 should be None'

    # transforms for view_3 and ...
    pre_transform3, post_transform3 = get_transforms(args.which_transform3, args.image_size)

    # dataset for all views
    from utils_data3 import MultiViewImageFolder
    sample_from_mixed = args.sample_from_mixed
    dataset = MultiViewImageFolder(
        args.data / 'train',
        index_filename='imagenet_index.pkl',
        latent_path=args.latent_path,
        view_paths=args.view_paths,
        transform0=pre_transform0,
        transform1=pre_transform1,
        transform2=pre_transform2,
        transform3=pre_transform3,
        n_views=args.n_views_gan,
        sample_from_mixed=sample_from_mixed,
        sample_from_original=args.sample_from_original,
    )
    return dataset, post_transform3


# rotation
def rotate_images(images, gpu, single=False):
    nimages = images.shape[0]
    
    if single:
        y = []
        for i in range(nimages):
            y.append(random.randint(0, 3))
            images[i] = torch.rot90(images[i], y[-1], [1, 2])
        y = torch.LongTensor(y).cuda()
        return images.cuda(gpu), y
    
    n_rot_images = 4 * nimages
    # rotate images all 4 ways at once
    rotated_images = torch.zeros(
        [n_rot_images, images.shape[1], images.shape[2], images.shape[3]]
        ).cuda(gpu, non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

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


def get_viewmaker(args, which_viewmaker='none'):
    if which_viewmaker == 'stnrand':
        from view_generator import RandomSpatialTransformerGenerator
        viewmaker = RandomSpatialTransformerGenerator()
        viewmaker_kwargs = {
            'init_scale': args.stn_init_scale,
            'noise_std': args.stn_noise_std,
        }
    else:
        viewmaker = lambda x, w, **kwargs: x
        viewmaker_kwargs = {}
    return viewmaker, viewmaker_kwargs


# main function
def main(parser):
    args = parser.parse_args()

    # ========== setup ==========
    if args.learning_rate is None:
        args.learning_rate = float(args.batch_size) ** 0.5 * 0.075
        print(f"learning_rate set to {args.learning_rate}")

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.log_dir = args.checkpoint_dir
    args.use_wandb = not args.no_wandb
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'weights'), exist_ok=True)

    print_args(parser, args)
    for filename in args.code_to_save.split(','):
        shutil.copy(filename, os.path.join(args.log_dir, 'src', filename+'.txt'))

    args.ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.distributed or args.ngpus_per_node > 1
    args.scale = [float(x) for x in args.scale.split(',') if x]
    if args.distributed:
        if 'SLURM_JOB_ID' in os.environ:
            cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
            stdout = subprocess.check_output(cmd.split())
            host_name = stdout.decode().splitlines()[0]
            args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
            args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
            args.dist_url = f'tcp://{host_name}:58478'
        else:
            # single-node distributed training
            args.rank = 0
            args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
            args.world_size = args.ngpus_per_node
        # NOTE: main_worker is called as main_worker(gpu_id, args), so *(args,) is args
        torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    else:
        args.rank = 0
        args.world_size = 1
        main_worker(0, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR Training')
    parser.add_argument('--data', type=Path, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate', default=None, type=float, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint-dir', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--rotation', default=0, type=float,
                        help="coefficient of rotation loss")
    parser.add_argument('--scale', default='0.05,0.14', type=str)

    # ========== add args ==========
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--cuda', default=None, type=str, help='cuda device ids to use')
    # parser.add_argument('--start_epoch', default=None, type=int, help='resume from this epoch')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='resume from this checkpoint')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='simclr-imagenet64', type=str)
    parser.add_argument('--code_to_save', type=str, default='utils_data.py',
        help='source files to save, note that main2.py will be saved in print_args')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--use_view_generator', action='store_true')
    parser.add_argument('--which_view_generator', default='none', type=str, choices=['none', 'gan', 'cache', 'stn', 'stnrand', 'stnrand2', 'stncrop'])
    parser.add_argument('--latent_path', type=str, default=None, help='dir root of latent files')
    parser.add_argument('--view_paths', type=str, default=None, help='dir root(s) of view files, comma separated')
    parser.add_argument('--sample_from_mixed', action='store_true')
    parser.add_argument('--sample_from_original', action='store_true')
    parser.add_argument('--n_views_gan', default=0, type=int, help='number of generated views, view_id >= 3')
    parser.add_argument('--which_transform0', type=str, default='resizedcrop+horizontalflip+colorjitter+grayscale+gaussianblur=1+solarization=0')
    parser.add_argument('--which_transform1', type=str, default='resizedcrop+horizontalflip+colorjitter+grayscale+gaussianblur=0.1+solarization=0.2')
    parser.add_argument('--which_transform2', type=str, default='resizedcrophalf+horizontalflip+colorjitter+grayscale+gaussianblur=0.1+solarization=0')
    parser.add_argument('--which_transform3', type=str, default='centercrop+horizontalflip+gan')

    parser.add_argument('--stn_init_scale', default=1., type=float)
    parser.add_argument('--stn_noise_std', default=0.1, type=float)
    # parser.add_argument('--stn_area_scale', type=str, default='0.2,1')
    # parser.add_argument('--stn_area_scale_clamp', action='store_true')
    # parser.add_argument('--stn_area_scale_strict', action='store_true')
    # parser.add_argument('--stn_aspect_ratio', type=str, default='0.75,1.333')
    parser.add_argument('--replace_expert_views', type=str, default=None, help="if '0,1', replace 0 and 1 with 3")
    parser.add_argument('--which_loss', type=str, default='simclr', choices=['simclr', 'simclr+pos'])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar100'])

    main(parser)
