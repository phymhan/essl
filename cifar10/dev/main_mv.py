import os
import math
import time
import copy
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T

from resnet import resnet18
from utils import knn_monitor, fix_seed, setup_wandb

import pdb
st = pdb.set_trace

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


class ContrastiveLearningTransform:
    def __init__(self, include_original=False, use_crop=False):
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
        self.transforms2 = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        ]) if use_crop else (lambda x: x)

        self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)
        self.include_original = include_original

    def __call__(self, x):
        output = [
            single_transform(x) if self.include_original else single_transform(self.transform(x)),
            single_transform(self.transform(x)),
            single_transform(self.transform_rotation(x))
        ]
        return output

class ContrastiveLearningTransform2:
    def __init__(self, use_crop0=False, use_crop1=False):
        if use_crop0:
            self.transform0 = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            ])
        else:
            self.transform0 = lambda x: x
        if use_crop1:
            self.transform1 = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform1 = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
            ])
        transforms_rotation = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        # self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)

    def __call__(self, x):
        output = [
            single_transform(self.transform0(x)),  # original image
            single_transform(self.transform1(x)),  # only random flip
            single_transform(self.transform_rotation(x))
        ]
        return output

class ContrastiveLearningTransform3:
    def __init__(self, use_crop0=False, use_crop1=False):
        if use_crop0:
            self.transform0 = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            ])
        else:
            self.transform0 = lambda x: x
        if use_crop1:
            self.transform1 = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform1 = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
            ])
        self.transforms = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ])
        transforms_rotation = [
            T.RandomResizedCrop(size=16, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        # self.transform = T.Compose(transforms)
        self.transform_rotation = T.Compose(transforms_rotation)

    def __call__(self, x):
        output = [
            single_transform(self.transform0(x)),  # original image
            single_transform(self.transforms(x)),  # expert
            single_transform(self.transform1(x)),  # only random flip
            single_transform(self.transform_rotation(x))
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


def info_nce_loss_mv(zs, temperature=0.5):
    n_views, batch_size = len(zs), zs[0].shape[0]

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    z = torch.nn.functional.normalize(torch.cat(zs, dim=0), dim=1)
    logits = z @ z.T
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    logits = logits[~mask].view(logits.shape[0], -1)
    positives = logits[labels.bool()].view(labels.shape[0], -1)
    negatives = logits[~labels.bool()].view(labels.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
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
            self.encoder = resnet18()
        self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )
        if args.loss == 'simclr':
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 4))  # output layer
        else:
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, 4))  # output layer

    def forward(self, x):
        return self.net(x)


def knn_loop(encoder, train_loader, test_loader):
    accuracy = knn_monitor(net=encoder.cuda(),
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
        saved_dict = torch.load(os.path.join(args.checkpoint_path))['state_dict']
        main_branch.load_state_dict(saved_dict)
        file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log.txt'), 'a')
        file_to_update.write(f'evaluating {args.checkpoint_path}\n')
        return main_branch.encoder, file_to_update

    total_step = 0

    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log.txt'), 'w')

    # dataset
    if args.use_view_generator:
        if args.use_pos_view:
            train_loader = torch.utils.data.DataLoader(
                dataset=torchvision.datasets.CIFAR10(
                    '../data', train=True,
                    transform=ContrastiveLearningTransform3(
                        use_crop0=args.use_crop0,
                        use_crop1=args.use_crop1,
                    ),
                    download=True
                ),
                shuffle=True,
                batch_size=args.bsz,
                pin_memory=True,
                num_workers=args.num_workers,
                drop_last=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=torchvision.datasets.CIFAR10(
                    '../data', train=True,
                    transform=ContrastiveLearningTransform2(
                        use_crop0=args.use_crop0,
                        use_crop1=args.use_crop1,
                    ),
                    download=True
                ),
                shuffle=True,
                batch_size=args.bsz,
                pin_memory=True,
                num_workers=args.num_workers,
                drop_last=True
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                '../data', train=True,
                transform=ContrastiveLearningTransform(args.include_original),
                download=True
            ),
            shuffle=True,
            batch_size=args.bsz,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=True
        )
    memory_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            '../data', train=True, transform=single_transform, download=True
        ),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            '../data', train=False, transform=single_transform, download=True,
        ),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # models

    main_branch = Branch(args, encoder=encoder).cuda()
    if args.use_view_generator:
        from view_generator import ViewGeneratorFGSM, get_gan_models
        if args.pretrained_encoder_path:
            viewmaker_encoder = Branch(args, encoder=encoder).cuda()
            saved_dict = torch.load(os.path.join(args.pretrained_encoder_path))['state_dict']
            viewmaker_encoder.load_state_dict(saved_dict)
            freeze_encoder = True
        else:
            viewmaker_encoder = main_branch
            freeze_encoder = False
        gan_gen, gan_enc = get_gan_models('cuda')
        viewmaker = ViewGeneratorFGSM(
            gan_generator=gan_gen,
            gan_encoder=gan_enc,
            simclr_encoder=viewmaker_encoder,
            idinvert_steps=args.idinvert_steps,
            boundary_steps=args.boundary_steps,
            boundary_epsilon=args.boundary_epsilon,
            fgsm_eps=args.fgsm_eps,
            freeze_encoder=freeze_encoder,
        )
    else:
        viewmaker = None

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

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))
    scaler = GradScaler()

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.loss == 'simsiam':
            predictor.train()

        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # synthesize views
            if args.use_view_generator:
                if args.use_pos_view:
                    inputs[2] = viewmaker(inputs[2].cuda())
                else:
                    inputs[1] = viewmaker(inputs[1].cuda())

            # adjust
            lr = adjust_learning_rate(epochs=args.epochs,
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
                b1 = backbone(x1)
                b2 = backbone(x2)
                z1 = projector(b1)
                z2 = projector(b2)

                # forward pass
                if args.loss == 'simclr':
                    loss = info_nce_loss(z1, z2) / 2 + info_nce_loss(z2, z1) / 2
                    if args.use_pos_view:
                        x3 = inputs[2]
                        b3 = backbone(x3)
                        z3 = projector(b3)
                        loss -= torch.sum(torch.nn.functional.normalize(z1, dim=1) * torch.nn.functional.normalize(z3, dim=1), dim=1).mean()
                elif args.loss == 'simsiam':
                    x1 = inputs[0].cuda()
                    x2 = inputs[1].cuda()
                    b1 = backbone(x1)
                    b2 = backbone(x2)
                    z1 = projector(b1)
                    z2 = projector(b2)
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
            f'time_elapsed: {time.time() - start:.3f}'
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
                    'time_elapsed': time.time() - start
                }
            )

        if e % args.save_every == 0:
            torch.save(dict(epoch=e, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))

    return main_branch.encoder, file_to_update


def eval_loop(encoder, file_to_update, ind=None, logger=None):
    # dataset
    train_transform = T.Compose([
        T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(32),
        T.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../data', train=True, transform=train_transform, download=True),
        shuffle=True,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../data', train=False, transform=test_transform, download=True),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers
    )

    classifier = nn.Linear(512, 10).cuda()
    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, 101):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=100,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(inputs.cuda())
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
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    return accuracy


def main(args):
    fix_seed(args.seed)
    logger = setup_wandb(args) if args.use_wandb else None
    encoder, file_to_update = ssl_loop(args, logger=logger)
    accs = []
    for i in range(5):
        accs.append(eval_loop(copy.deepcopy(encoder), file_to_update, i))
    line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)
    if logger is not None:
        import wandb
        tbl = wandb.Table(data=[accs + [np.mean(accs)]], columns=[f'seed={i}' for i in range(5)] + ['mean'])
        logger.log({'linear_probe': tbl})


if __name__ == '__main__':
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
    parser.add_argument('--path_dir', default='../experiment', type=str)
    parser.add_argument('--wandb_project', default='simclr-cifar10', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lmbd', default=0.0, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--include_original', action='store_true')
    parser.add_argument('--use_view_generator', action='store_true')
    parser.add_argument('--pretrained_encoder_path', default=None, type=str)
    parser.add_argument('--fgsm_eps', default=0.1, type=float)
    parser.add_argument('--idinvert_steps', default=0, type=int)
    parser.add_argument('--boundary_steps', default=1, type=int)
    parser.add_argument('--boundary_epsilon', default=0.5, type=float)
    parser.add_argument('--use_crop0', action='store_true')
    parser.add_argument('--use_crop1', action='store_true')
    parser.add_argument('--use_pos_view', action='store_true')
    parser.add_argument('--use_pos_view_and_crop', action='store_true')
    args = parser.parse_args()

    args.log_dir = args.path_dir  # TODO

    main(args)
