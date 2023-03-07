import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps

import random

from .cars import Cars
from .synthetic import get_synthetic
from .airplane import Aircraft

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformN:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp,n=5):

        out = []
        for i in range(n):
            out.append(self.transform(inp))
        return out



def create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst, num_cls, ratio):
    np.random.seed(42)
    samples_per_class = np.zeros(num_cls)
    val_samples_per_class = np.zeros(num_cls)
    tst_samples_per_class = np.zeros(num_cls)
    for i in range(num_cls):
        samples_per_class[i] = len(np.where(y_trn == i)[0])
        val_samples_per_class[i] = len(np.where(y_val == i)[0])
        tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
    min_samples = int(np.min(samples_per_class) * 0.1)
    selected_classes = np.random.choice(np.arange(num_cls), size=int(ratio * num_cls), replace=False)
    for i in range(num_cls):
        if i == 0:
            if i in selected_classes:
                subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = x_trn[subset_idxs]
            y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
        else:
            if i in selected_classes:
                subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
            y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
    max_samples = int(np.max(val_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_val == i)[0]
        if i == 0:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val, x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1), y_val[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val, x_val_new, x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1), y_val_new, y_val[subset_ids].reshape(-1, 1)))
    max_samples = int(np.max(tst_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_tst == i)[0]
        if i == 0:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst, x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1), y_tst[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst, x_tst_new, x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1), y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

    return x_trn_new, y_trn_new.reshape(-1), x_val_new, y_val_new.reshape(-1), x_tst_new, y_tst_new.reshape(-1)


def create_noisy(y_trn, num_cls, noise_ratio=0.8):
    noise_size = int(len(y_trn) * noise_ratio)
    noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
    y_trn[noise_indices] = torch.from_numpy(np.random.choice(np.arange(num_cls), size=noise_size, replace=True))
    return y_trn, noise_indices


def load_dataset_custom(datadir, dset_name, feature, valid=True, isnumpy=False, \
                        transform2=False,transformN=False, **kwargs):
    if feature == 'classimb':
        if 'classimb_ratio' in kwargs:
            pass
        else:
            raise KeyError("Specify a classimb ratio value in the config file")

    if feature == 'noise':
        if 'noise_ratio' in kwargs:
            pass
        else:
            raise KeyError("Specify a noise ratio value in the config file")

    if dset_name == "synthetic":

        if 'type' in kwargs and 'test_type' in kwargs:
            pass
        else:
            raise KeyError("Specify a type and test type")

        fullset, testset, num_cls = get_synthetic(kwargs['seed'], kwargs['ncls'], kwargs['type'], kwargs['test_type'])#,transform=pertub)

        if valid:
            validation_set_fraction = 0.1
            num_fulltrn = len(fullset)
            num_val = int(num_fulltrn * validation_set_fraction)
            num_trn = num_fulltrn - num_val
            trainset, valset = random_split(fullset, [num_trn, num_val])

            if feature == 'noise':
                print((trainset.dataset[trainset.indices])[1][:20])
                fullset.targets[trainset.indices],_ = create_noisy((trainset.dataset[trainset.indices])[1], \
                                                                 num_cls, noise_ratio=kwargs['noise_ratio'])
                print((trainset.dataset[trainset.indices])[1][:20])
            return trainset, valset, testset, num_cls
        else:

            return fullset, testset, num_cls

    elif dset_name == "cifar100":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        crop_size=32
        cifar100_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        cifar100_tst_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        num_cls = 100

        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        testset = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True,
                                                transform=cifar100_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(cifar100_transform)
            fullset2 = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True,
                                                     transform=transform_train2)

        if transformN:
            transform_trainN = TransformN(cifar100_transform)
            fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True,
                                                     transform=transform_trainN)

        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls),
                                                replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        if valid:
            validation_set_fraction = 0.1
            num_fulltrn = len(fullset)
            num_val = int(num_fulltrn * validation_set_fraction)
            num_trn = num_fulltrn - num_val
            trainset, valset = random_split(fullset, [num_trn, num_val])


            if feature == 'noise':
                tensor_target = torch.tensor(fullset.targets)
                tensor_target[trainset.indices],noise_idx = create_noisy((tensor_target[trainset.indices]), \
                                                                 num_cls, noise_ratio=kwargs['noise_ratio'])

                fullset.targets = tensor_target.tolist()
                
            if transform2:
                trainset2, _ = random_split(fullset2, [num_trn, num_val])
                return trainset2, valset, testset, num_cls
            else:
                if feature == 'noise':
                    return trainset, valset, testset, num_cls,noise_idx
                else:
                    return trainset, valset, testset, num_cls
                #return fullset, valset, tstset, num_cls
        else:
            if transform2:
                return fullset, fullset2, testset, num_cls
            else:
                return fullset, testset, num_cls

    elif dset_name == 'airplane':

        crop_size = 96 #224
        transform_train = transforms.Compose([transforms.Resize(crop_size),transforms.RandomCrop(crop_size),\
        transforms.RandomRotation(45),transforms.RandomHorizontalFlip(),transforms.ToTensor(),\
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([transforms.Resize(crop_size),transforms.RandomCrop(crop_size),\
        transforms.ToTensor(),torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        num_cls = 102

        fullset = Aircraft(root=datadir, split='trainval', download=True, transform=transform_train)
        testset = Aircraft(root=datadir, split='test', download=True, transform=transform_test)

        if valid:
            validation_set_fraction = 0.1
            num_fulltrn = len(fullset)
            num_val = int(num_fulltrn * validation_set_fraction)
            num_trn = num_fulltrn - num_val
            trainset, valset = random_split(fullset, [num_trn, num_val])
            
            return trainset, valset, testset, num_cls
        else:
            return fullset, testset, num_cls

    elif dset_name == 'cars':

        crop_size = 96 #224 #96
        transform_train = transforms.Compose([transforms.Resize(crop_size),transforms.RandomCrop(crop_size),\
        transforms.RandomRotation(45),transforms.RandomHorizontalFlip(),transforms.ToTensor(),\
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([transforms.Resize(crop_size),transforms.RandomCrop(crop_size),\
        transforms.ToTensor(),torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        num_cls = 196

        fullset = Cars(root=datadir, train=True, download=True, transform=transform_train)
        testset = Cars(root=datadir, train=False, download=True, transform=transform_test)

        if transform2:
            transform_train2 = TransformTwice(transform_train)
            fullset2 = Cars(root=datadir, train=True, download=True, transform=transform_train2)

        if valid:
            validation_set_fraction = 0.1
            num_fulltrn = len(fullset)
            num_val = int(num_fulltrn * validation_set_fraction)
            num_trn = num_fulltrn - num_val
            trainset, valset = random_split(fullset, [num_trn, num_val])

            if transform2:
                trainset2, _ = random_split(fullset2, [num_trn, num_val])
                return trainset, trainset2, valset, testset, num_cls
            else:
                return trainset, valset, testset, num_cls
        else:
            if transform2:
                return fullset, fullset2, testset, num_cls
            else:
                return fullset, testset, num_cls

    