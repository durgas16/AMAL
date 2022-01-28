import os
import torch
from torch.utils.data import Dataset, random_split

import sklearn.datasets as dt
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            self.targets = target.long().to(device)
        else:
            self.data = data.float()
            self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return [sample_data, label]  # .astype('float32')


def get_synthetic(seed=42, classes=4, type="class", test_type="None", noise_ratio=0, cls_ratio=None,transform=None):
    np.random.seed(seed)

    a = np.array([0.25 * i for i in range(1, 10 * classes + 1)])
    ind = np.random.permutation(10 * classes)

    if type == "class":
        if cls_ratio is None:
            if test_type == "Imb":
                cls_ratio = a[ind[:classes]] / np.sum(a[ind[:classes]])
            else:
                cls_ratio = [1 / classes for _ in range(classes)]

        if test_type == "Noise":
            noise_ratio = 0.2

        x, y = dt.make_classification(n_samples=10000,
                                      n_features=14,
                                      n_informative=14,
                                      n_repeated=0,
                                      class_sep=1.0,
                                      n_classes=classes,
                                      n_redundant=0,
                                      n_clusters_per_class=1,
                                      random_state=seed,
                                      flip_y=noise_ratio,
                                      weights=cls_ratio)
    elif type == "spiral":
        N = 1500  # number of points per class
        D = 2  # dimensionality
        K = classes  # number of classes
        x = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 4, N)  # radius
            t = np.linspace(j * 4.74, (j + 1) * 4.74, N) + np.random.randn(N) * 0.2  # theta
            x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
    elif type == "blob":

        if cls_ratio is None:
            if test_type == "Imb":
                cls_ratio = 10000 * a[ind[:classes]] / np.sum(a[ind[:classes]])
            else:
                cls_ratio = [int(10000 / classes) for _ in range(classes)]

        '''x, y = dt.make_blobs(n_samples=10000,
                             n_features=2,
                             centers=classes,
                             random_state=seed)  # cluster_std=a[ind[:classes]],'''
        x, y = dt.make_blobs(n_samples=10000, centers=8, center_box=(- 20.0, 20.0),
                               cluster_std=[1.4, 2.8, 1.7, 4, 2, 3.5, 1.5, 2.5], random_state=20)

        if test_type == "Noise":
            train_x, train_y = torch.tensor(x[:int(len(x) * 0.9)]), torch.tensor(y[:int(len(y) * 0.9)])

            noise_size = int(len(train_y) * noise_ratio)
            noise_indices = np.random.choice(np.arange(len(train_y)), size=noise_size, replace=False)
            train_y[noise_indices] = torch.tensor(np.random.choice(np.arange(classes), size=noise_size, replace=True))
            train = CustomDataset(train_x, train_y)

            test = CustomDataset(torch.tensor(x[int(len(x) * 0.9):]), torch.tensor(y[int(len(y) * 0.9):]))
            return train, test, classes

    '''init_dataset = CustomDataset(torch.tensor(x),torch.tensor(y))
    lengths = [int(len(init_dataset)*0.9), int(len(init_dataset)*0.1)]
 
    train, test = random_split(init_dataset, lengths)'''

    '''if test_type == "Noise":
        train_x, train_y = torch.tensor(x[:int(len(x)*0.9)]),torch.tensor(y[:int(len(y)*0.9)])
 
        noise_size = int(len(train_y) * noise_ratio)
        noise_indices = np.random.choice(np.arange(len(train_y)), size=noise_size, replace=False)
        train_y[noise_indices] = torch.tensor(np.random.choice(np.arange(classes), size=noise_size, replace=True))
        train = CustomDataset(train_x, train_y)
 
        test = CustomDataset(torch.tensor(x[int(len(x)*0.9):]),torch.tensor(y[int(len(y)*0.9):]))
    elif test_type == "Shift":
        train = CustomDataset(torch.tensor(x[:int(len(x)*0.9)]),torch.tensor(y[:int(len(y)*0.9)]))
        test_x, test_y = torch.tensor(x[int(len(x)*0.9):]),torch.tensor(y[int(len(y)*0.9):])
        test_x[:0] += np.random.random_sample()*5
        test_x[:1] += np.random.random_sample()*5
 
        test = CustomDataset(test_x, test_y)
    else:'''

    sh_x = torch.randperm(x.shape[0])
    
    train_ind = sh_x[:int(len(x) * 0.9)]
    '''noise_size=int(len(train_ind)*0.2)
    noise_indices = np.random.choice(train_ind, size=noise_size, replace=False)
    y[noise_indices] = torch.tensor(np.random.choice(np.arange(classes), size=noise_size, replace=True))'''

    train = CustomDataset(torch.tensor(x[train_ind]), torch.tensor(y[train_ind]), transform=transform)
    test = CustomDataset(torch.tensor(x[sh_x[int(len(x) * 0.9):]]), torch.tensor(y[sh_x[int(len(y) * 0.9):]]),transform=transform)

    return train, test, classes




