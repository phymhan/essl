import os
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as T
# from torch.utils.data.dataset import Subset
# from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)

import pdb
st = pdb.set_trace


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path='data.pkl',
        latent_path='latent.pkl',
        view_paths=['view.pkl'],
        transform0=None,
        transform1=None,
        transform2=None,
        transform3=None,
        n_views=1,
        train=True,
        subset_index=None,
        sample_from_mixed=False,
        sample_original=False,
    ):
        """
        images: [x0, x1, x2, x3, ...]
            x0, x1 are two expert views
            x2 are rotation views reserved for essl
            x3 and after are for (cached) GAN generated views
        returns: (images, latents, labels)
        """
        # self.data_transform = lambda x: (x + 1) / 2  # NOTE: to [0, 1]
        self.transform0 = transform0
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        # load data
        assert os.path.exists(data_path), f'data_path {data_path} does not exist'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        # load latent
        assert os.path.exists(latent_path), f'latent_path {latent_path} does not exist'
        with open(latent_path, 'rb') as f:
            latents = pickle.load(f)
        if subset_index is None:
            if 'index' in data:
                subset_index = data['index']['train'] if train else data['index']['test']
            else:
                subset_index = np.arange(len(data['images']))
        self.data = {
            'images': data['images'][subset_index],
            'labels': torch.tensor(data['labels'])[subset_index],
            'latents': latents['latents'][subset_index],
        }
        # load views
        if isinstance(view_paths, str):
            view_paths = [view_paths]
        views = []
        for view_path in view_paths:
            assert os.path.exists(view_path), f'view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, subset_index])
        # NOTE: original data is view_0 (this is convenient if we want to sample from mixed pool)
        self.views = torch.cat([self.data['images'].unsqueeze(0)] + views, dim=0)
        self.total_views = self.views.shape[0]  # NOTE: including original data view_0
        self.n_views = n_views  # NOTE: number of generated views to sample
        assert sample_original or self.n_views <= self.total_views - int(not sample_from_mixed)
        self.sample_from_mixed = sample_from_mixed
        self.sample_original = sample_original
        self.dataset_len = len(subset_index)
    
    def data_transform(self, image_tensor):
        image_tensor = (image_tensor + 1) / 2
        return image_tensor

    def __getitem__(self, index):
        # view_0
        img0 = self.data_transform(self.data['images'][index])
        if self.transform0 is not None:
            img0 = self.transform0(img0)
        # view_1
        img1 = self.data_transform(self.data['images'][index])
        if self.transform1 is not None:
            img1 = self.transform1(img1)
        # view_2
        img2 = self.data_transform(self.data['images'][index])
        if self.transform2 is not None:
            img2 = self.transform2(img2)
        # view_3 and ...
        views = []
        if self.sample_original:
            inds = [0] * self.n_views
        else:
            if self.sample_from_mixed:
                inds = list(np.random.choice(np.arange(0, self.total_views), size=self.n_views, replace=False))
            else:
                inds = list(np.random.choice(np.arange(1, self.total_views), size=self.n_views, replace=False))
        for i in inds:
            img = self.data_transform(self.views[i, index])
            if self.transform3 is not None:
                img = self.transform3(img)
            views.append(img)
        # latent
        latents = self.data['latents'][index]
        # label
        labels = self.data['labels'][index]
        images = [img0, img1, img2] + views
        return images, latents, labels

    def __len__(self):
        return self.dataset_len


class PosNegViewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path='data.pkl',
        latent_path='latent.pkl',
        pos_view_paths=['view_pos.pkl'],
        neg_view_paths=['view_neg.pkl'],
        transform0=None,
        transform1=None,
        transform2=None,
        transform3=None,
        n_views=1,
        train=True,
        subset_index=None,
        sample_from_mixed=False,
        sample_original=False,
    ):
        """
        images: [x0, x1, x2, x3, ...]
            x0, x1 are two expert views
            x2 are rotation views reserved for essl
            x3 and after are for (cached) GAN generated views
        returns: (images, latents, labels)
        """
        # self.data_transform = lambda x: (x + 1) / 2  # NOTE: to [0, 1]
        self.transform0 = transform0
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        # load data
        assert os.path.exists(data_path), f'data_path {data_path} does not exist'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        # load latent
        assert os.path.exists(latent_path), f'latent_path {latent_path} does not exist'
        with open(latent_path, 'rb') as f:
            latents = pickle.load(f)
        if subset_index is None:
            if 'index' in data:
                subset_index = data['index']['train'] if train else data['index']['test']
            else:
                subset_index = np.arange(len(data['images']))
        self.data = {
            'images': data['images'][subset_index],
            'labels': torch.tensor(data['labels'])[subset_index],
            'latents': latents['latents'][subset_index],
        }
        # load views

        if isinstance(pos_view_paths, str):
            pos_view_paths = [pos_view_paths]
        views = []
        for view_path in pos_view_paths:
            assert os.path.exists(view_path), f'pos_view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, subset_index])
        # NOTE: original data is view_0 (this is convenient if we want to sample from mixed pool)
        self.views_pos = torch.cat([self.data['images'].unsqueeze(0)] + views, dim=0)
        self.total_views_pos = self.views_pos.shape[0]  # NOTE: including original data view_0

        if isinstance(neg_view_paths, str):
            neg_view_paths = [neg_view_paths]
        views = []
        for view_path in neg_view_paths:
            assert os.path.exists(view_path), f'neg_view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, subset_index])
        # NOTE: original data is view_0 (this is convenient if we want to sample from mixed pool)
        self.views_neg = torch.cat([self.data['images'].unsqueeze(0)] + views, dim=0)
        self.total_views_neg = self.views_neg.shape[0]  # NOTE: including original data view_0

        # NOTE: n_views is number of views for both pos and neg
        assert n_views % 2 == 0, f'n_views should be even, but got {n_views}'
        n_views = n_views // 2
        self.n_views = n_views  # NOTE: number of generated views to sample
        assert sample_original or self.n_views <= self.total_views_pos - int(not sample_from_mixed)
        assert sample_original or self.n_views <= self.total_views_neg - int(not sample_from_mixed)
        self.sample_from_mixed = sample_from_mixed
        self.sample_original = sample_original
        self.dataset_len = len(subset_index)
    
    def data_transform(self, image_tensor):
        image_tensor = (image_tensor + 1) / 2
        return image_tensor

    def __getitem__(self, index):
        # view_0
        img0 = self.data_transform(self.data['images'][index])
        if self.transform0 is not None:
            img0 = self.transform0(img0)
        # view_1
        img1 = self.data_transform(self.data['images'][index])
        if self.transform1 is not None:
            img1 = self.transform1(img1)
        # view_2
        img2 = self.data_transform(self.data['images'][index])
        if self.transform2 is not None:
            img2 = self.transform2(img2)
        # view_3 and ...
        views = []
        if self.sample_original:
            inds_pos = [0] * self.n_views
            inds_neg = [0] * self.n_views
        else:
            if self.sample_from_mixed:
                inds_pos = list(np.random.choice(np.arange(0, self.total_views_pos), size=self.n_views, replace=False))
                inds_neg = list(np.random.choice(np.arange(0, self.total_views_neg), size=self.n_views, replace=False))
            else:
                inds_pos = list(np.random.choice(np.arange(1, self.total_views_pos), size=self.n_views, replace=False))
                inds_neg = list(np.random.choice(np.arange(1, self.total_views_neg), size=self.n_views, replace=False))
        for i in inds_pos:
            img = self.data_transform(self.views_pos[i, index])
            if self.transform3 is not None:
                img = self.transform3(img)
            views.append(img)
        for i in inds_neg:
            img = self.data_transform(self.views_neg[i, index])
            if self.transform3 is not None:
                img = self.transform3(img)
            views.append(img)
        # latent
        latents = self.data['latents'][index]
        # label
        labels = self.data['labels'][index]
        images = [img0, img1, img2] + views
        return images, latents, labels

    def __len__(self):
        return self.dataset_len
