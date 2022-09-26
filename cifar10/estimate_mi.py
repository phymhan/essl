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
    parser.add_argument('--which_pair', type=str, default='0,1')
    parser.add_argument('--uint8', action='store_true')
    parser.add_argument('--mi_model_name', type=str, default='mine')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--transform2', type=str, default='expert')
    args = parser.parse_args()

    device = 'cuda'

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
        expert_transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        image_size = 32
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
        expert_transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        image_size = 32
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
        expert_transform = T.Compose([
            T.RandomResizedCrop(size=64, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        image_size = 64
    
    normalize = T.Normalize(mean=mean, std=std)
    # single_transform = T.Compose([T.ToTensor(), normalize])
    single_transform = T.Compose([T.ToTensor(), normalize])
    small_transform = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.9, 1.), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=mean, std=std),
    ])
    
    model.eval()  # NOTE

    # === define MI estimator ===
    model = None
    # from club.mi_estimators import ConvCLUBSample
    # estimator = ConvCLUBSample().to(device)
    from club.mi_estimators import ConvMINE
    estimator = ConvMINE().to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), 0.005)


    batch_size = 256

    from utils_data import MultiViewDataset
    # train_dataset = MultiViewDataset(
    #     data_path=args.data_path,
    #     latent_path=args.latent_path,
    #     view_paths=[args.view_path],
    #     transform0=normalize,
    #     transform1=normalize,
    #     transform2=normalize,
    #     transform3=normalize,
    #     n_views=2,  # NOTE: we use 0, 3, 4
    #     train=True,
    # )
    if args.transform2 == 'expert':
        transform2 = expert_transform
    elif args.transform2 == 'small':
        transform2 = small_transform
    elif args.transform2 == 'flip':
        transform2 = T.Compose([normalize, T.RandomHorizontalFlip(0.5)])
    train_dataset = MultiViewDataset(
        data_path=args.data_path,
        latent_path=args.latent_path,
        view_paths=[args.view_path],
        transform0=T.Compose([normalize, T.RandomHorizontalFlip()]) if args.flip else normalize,
        transform1=expert_transform,
        transform2=transform2,
        transform3=normalize,
        n_views=2,  # NOTE: we use 0, 1, 3
        train=True,
        uint8=args.uint8,
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

    loader = sample_data(train_loader)

    estimator.train()

    # with torch.no_grad():
    for e in tqdm(range(args.epochs)):
        mi_est_values = []
        pbar = tqdm(train_loader)
        for imgs, _, _ in pbar:
            # imgs, _, _ = next(loader)

            if args.which_pair == '0,1':
                batch_x = imgs[0].to(device)
                batch_y = imgs[1].to(device)
            # elif args.which_pair == '0,0':
            #     batch_x = imgs[0].to(device)
            #     batch_y = imgs[0].to(device)
            elif args.which_pair == '0,2':
                batch_x = imgs[0].to(device)
                batch_y = imgs[2].to(device)
            elif args.which_pair == '1,2':
                batch_x = imgs[1].to(device)
                batch_y = imgs[2].to(device)
            elif args.which_pair == '0,3':
                batch_x = imgs[0].to(device)
                batch_y = imgs[3].to(device)
            elif args.which_pair == '1,3':
                batch_x = imgs[1].to(device)
                batch_y = imgs[3].to(device)
            elif args.which_pair == '3,1':
                batch_x = imgs[3].to(device)
                batch_y = imgs[1].to(device)
            elif args.which_pair == '3,4':
                batch_x = imgs[3].to(device)
                batch_y = imgs[4].to(device)

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
    for imgs, _, _ in pbar:
        if args.which_pair == '0,1':
            batch_x = imgs[0].to(device)
            batch_y = imgs[1].to(device)
        # elif args.which_pair == '0,0':
        #     batch_x = imgs[0].to(device)
        #     batch_y = imgs[0].to(device)
        elif args.which_pair == '0,2':
            batch_x = imgs[0].to(device)
            batch_y = imgs[2].to(device)
        elif args.which_pair == '1,2':
            batch_x = imgs[1].to(device)
            batch_y = imgs[2].to(device)
        elif args.which_pair == '0,3':
            batch_x = imgs[0].to(device)
            batch_y = imgs[3].to(device)
        elif args.which_pair == '1,3':
            batch_x = imgs[1].to(device)
            batch_y = imgs[3].to(device)
        elif args.which_pair == '3,1':
            batch_x = imgs[3].to(device)
            batch_y = imgs[1].to(device)
        elif args.which_pair == '3,4':
            batch_x = imgs[3].to(device)
            batch_y = imgs[4].to(device)

        # estimator.eval()
        m = estimator(batch_x, batch_y).item()
        mi_est_values.append(m)
        # pbar.set_description('MI Estimator: %.4f' % m)
    print(f"MI Estimator: {np.mean(mi_est_values):.4f}")
    # torch.save(estimator.state_dict(), f"{args.mi_model_name}_{args.dataset}.pt")
