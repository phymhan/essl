from ast import Mult
import os
import random
from re import X
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
import pickle
import pdb
st = pdb.set_trace


class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class ViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, view_paths):
        self.dataset = dataset
        self.n_cache = 8
        if isinstance(view_paths, str):
            view_paths = [view_paths]
        views = []
        for view_path in view_paths:
            assert os.path.exists(view_path), f'view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:self.n_cache, np.arange(50000)])  # NOTE: hardcoded
        self.views = torch.cat(views, dim=0)
        self.views = (self.views + 1) / 2
        self.view_transform = self.dataset.transform.view_transform
    
    def __getitem__(self, index):
        (img1, img2), y = self.dataset[index]
        inds = np.random.choice(np.arange(0, self.n_cache), size=1, replace=False)[0]
        img3 = self.views[inds, index]
        img3 = self.view_transform(img3)
        return (img1, img2, img3), y
    
    def __len__(self):
        return len(self.dataset)

class ContrastiveLearningTransformCIFAR10:
    def __init__(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.view_transform = normalize
        self.expert_transform = T.Compose([
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
            self.expert_transform(x),
            # self.view_transform(x),
        ]
        return output

class ContrastiveLearningTransformCIFAR100:
    def __init__(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.view_transform = normalize
        self.expert_transform = T.Compose([
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
            self.expert_transform(x),
            # self.view_transform(x),
        ]
        return output

class ContrastiveLearningTransformTinyImageNet1:
    def __init__(self):
        mean = (0.480, 0.448, 0.398)
        std = (0.277, 0.269, 0.282)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.view_transform = normalize
        self.expert_transform = T.Compose([
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
            self.expert_transform(x),
            # self.view_transform(x),
        ]
        return output

class ContrastiveLearningTransformTinyImageNet2:
    def __init__(self):
        mean = (0.480, 0.448, 0.398)
        std = (0.277, 0.269, 0.282)
        normalize = T.Normalize(mean=mean, std=std)
        self.single_transform = T.Compose([T.ToTensor(), normalize])
        self.view_transform = normalize
        self.expert_transform = T.Compose(
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
            self.expert_transform(x),
            # self.view_transform(x),
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
    parser.add_argument('--which_pair', type=str, default='0,1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--view_path', type=str, default='data/c10_obj=l2_eps=0.3_loss=l2_reg=0.01_eps2=0.5_n=8_seed=0.pkl')
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
        zdim = 2048
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
        zdim = 128
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
        zdim = 128
    
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
        zdim = 128
    
    if not args.no_eval:
        model.eval()  # NOTE
    
    from club.mi_estimators import MINE
    estimator = MINE(zdim, zdim, zdim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), 0.005)

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
    
    train_dataset = ViewDataset(train_dataset, args.view_path)

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

    # loader = sample_data(train_loader)
    estimator.train()
    for e in tqdm(range(args.epochs)):
        mi_est_values = []
        pbar = tqdm(train_loader)
        for imgs, _ in pbar:
            # imgs, _, _ = next(loader)
            if args.which_pair == '0,1':
                batch_x = imgs[0].to(device)
                batch_y = imgs[1].to(device)
            elif args.which_pair == '0,2':
                batch_x = imgs[0].to(device)
                batch_y = imgs[2].to(device)
            elif args.which_pair == '1,2':
                batch_x = imgs[1].to(device)
                batch_y = imgs[2].to(device)
            elif args.which_pair == '2,1':
                batch_x = imgs[2].to(device)
                batch_y = imgs[1].to(device)
            batch_x = model(batch_x)
            batch_y = model(batch_y)
            batch_x = F.normalize(batch_x, dim=1)
            batch_y = F.normalize(batch_y, dim=1)

            estimator.eval()
            m = estimator(batch_x, batch_y).item()
            mi_est_values.append(m)
            pbar.set_description('MI Estimator: %.4f' % m)

            estimator.train()
            mi_loss = estimator.learning_loss(batch_x, batch_y)
            optimizer.zero_grad()
            mi_loss.backward()
            optimizer.step()

        if e % 10 == 0:
            # print('pair ', args.which_pair, np.mean(mi_est_values))
            print(f"MI Estimator: {np.mean(mi_est_values):.4f}")
    
    estimator.eval()
    mi_est_values = []
    pbar = tqdm(train_loader)
    for imgs, _ in pbar:
        # imgs, _, _ = next(loader)
        if args.which_pair == '0,1':
            batch_x = imgs[0].to(device)
            batch_y = imgs[1].to(device)
        elif args.which_pair == '0,2':
            batch_x = imgs[0].to(device)
            batch_y = imgs[2].to(device)
        elif args.which_pair == '1,2':
            batch_x = imgs[1].to(device)
            batch_y = imgs[2].to(device)
        elif args.which_pair == '2,1':
            batch_x = imgs[2].to(device)
            batch_y = imgs[1].to(device)
        batch_x = model(batch_x)
        batch_y = model(batch_y)
        batch_x = F.normalize(batch_x, dim=1)
        batch_y = F.normalize(batch_y, dim=1)

        m = estimator(batch_x, batch_y).item()
        mi_est_values.append(m)
        # pbar.set_description('MI Estimator: %.4f' % m)
    print(f"MI Estimator: {np.mean(mi_est_values):.4f}")
    # torch.save(estimator.state_dict(), f"{args.mi_model_name}_{args.dataset}.pt")
