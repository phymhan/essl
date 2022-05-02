import os
# import imageio
# import yaml
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)
import pickle
import numpy as np
import pdb
st = pdb.set_trace


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data.pkl', view_paths=['view.pkl'], train=True, transform=None, n_views=2,
                 must_include_original=False):
        # TODO: hardcoded
        if train:
            index = np.arange(50000)
        else:
            index = np.arange(50000, 60000)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'{data_path} does not exist')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data_path = data_path
        self.data = {
            'images': data['images'][index],
            'labels': torch.tensor(data['labels'])[index],
        }
        views = []
        if isinstance(view_paths, str):
            view_paths = [view_paths]
        for view_path in view_paths:
            if not os.path.exists(view_path):
                raise FileNotFoundError(f'{view_path} does not exist')
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, index])
        self.view_paths = view_paths
        # NOTE: original data is view_0
        self.views = torch.cat([self.data['images'].unsqueeze(0)] + views, dim=0)
        # self.total_views = self.views.shape[0] - 1
        self.total_views = self.views.shape[0]
        self.n_views = n_views
        self.transform = transform
        self.must_include_original = must_include_original

    def __getitem__(self, index):
        if self.total_views == 0:
            inds = [0] * self.n_views
        else:
            if self.must_include_original:
                inds = [0] + list(np.random.choice(np.arange(1, self.total_views), self.n_views - 1, replace=False))
            else:
                inds = np.random.choice(np.arange(self.total_views), self.n_views, replace=False)
        imgs = []
        for i in inds:
            img = self.views[i, index].clone()
            img = (img + 1) / 2  # main assumes range is [0, 1]
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        lbls = self.data['labels'][index]
        return imgs, lbls

    def __len__(self):
        return len(self.data['labels'])


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data.pkl', latent_path='latent.pkl', train=True, transform=None):
        # TODO: hardcoded
        if train:
            index = np.arange(50000)
        else:
            index = np.arange(50000, 60000)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'{data_path} does not exist')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data_path = data_path
        self.data = {
            'images': data['images'][index],
            'labels': torch.tensor(data['labels'])[index],
        }
        with open(latent_path, 'rb') as f:
            latents = pickle.load(f)
        self.latent_path = latent_path
        self.data['latents'] = latents['latents'][index]
        self.transform = transform

    def __getitem__(self, index):
        x = self.data['images'][index].clone()
        w = self.data['latents'][index].clone()
        y = self.data['labels'][index]

        if self.transform is not None:
            x = self.transform((x + 1) / 2)

        return x, w, y

    def __len__(self):
        return len(self.data['labels'])


class PosViewDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data.pkl', view_paths=['view.pkl'], train=True, transform01=None, transform2=None, n_views=2,
                 must_include_original=False, transform0=None):
        # TODO: hardcoded
        if train:
            index = np.arange(50000)
        else:
            index = np.arange(50000, 60000)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'{data_path} does not exist')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data_path = data_path
        self.data = {
            'images': data['images'][index],
            'labels': torch.tensor(data['labels'])[index],
        }
        views = []
        if isinstance(view_paths, str):
            view_paths = [view_paths]
        for view_path in view_paths:
            if not os.path.exists(view_path):
                raise FileNotFoundError(f'{view_path} does not exist')
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, index])
        self.view_paths = view_paths
        # NOTE: original data is view_0
        self.views = torch.cat([self.data['images'].unsqueeze(0)] + views, dim=0)
        self.total_views = self.views.shape[0]
        self.n_views = n_views
        self.transform01 = transform01
        self.transform2 = transform2
        self.must_include_original = must_include_original
        self.transform0 = transform0

    def __getitem__(self, index):
        if self.transform01 is not None:
            if self.must_include_original:
                imgs = [
                    self.transform0((self.data['images'][index].clone() + 1) / 2),
                    self.transform01((self.data['images'][index].clone() + 1) / 2)
                ]
            else:
                imgs = [
                    self.transform01((self.data['images'][index].clone() + 1) / 2),
                    self.transform01((self.data['images'][index].clone() + 1) / 2)
                ]
        else:
            raise ValueError('transform01 is None')
            imgs = [(self.data['images'][index].clone() + 1) / 2, (self.data['images'][index].clone() + 1) / 2]
        inds = list(np.random.choice(np.arange(1, self.total_views), self.n_views, replace=False))
        for i in inds:
            img = self.views[i, index].clone()
            img = (img + 1) / 2  # main assumes range is [0, 1]
            if self.transform2 is not None:
                img = self.transform2(img)
            imgs.append(img)
        lbls = self.data['labels'][index]
        return imgs, lbls

    def __len__(self):
        return len(self.data['labels'])