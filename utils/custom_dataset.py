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

from wilds.common.data_loaders import get_train_loader
from wilds import get_dataset

import random

from .cars import Cars
from .synthetic import get_synthetic

import medmnist
from medmnist import INFO, Evaluator


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
    return y_trn


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
                fullset.targets[trainset.indices] = create_noisy((trainset.dataset[trainset.indices])[1], \
                                                                 num_cls, noise_ratio=kwargs['noise_ratio'])
                print((trainset.dataset[trainset.indices])[1][:20])
            return trainset, valset, testset, num_cls
        else:

            return fullset, testset, num_cls


    elif dset_name == "mnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_tst_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10

        fullset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform)
        testset = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_tst_transform)
        if transform2:
            transform_train2 = TransformTwice(mnist_transform)
            fullset2 = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=transform_train2)

        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls),
                                                replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
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


    elif dset_name == "fashion-mnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_tst_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10

        fullset = torchvision.datasets.FashionMNIST(root=datadir, train=True, download=True,
                                                    transform=mnist_transform)
        testset = torchvision.datasets.FashionMNIST(root=datadir, train=False, download=True,
                                                    transform=mnist_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(mnist_transform)
            fullset2 = torchvision.datasets.FashionMNIST(root=datadir, train=True, download=True,
                                                         transform=transform_train2)

        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls),
                                                replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
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

    elif dset_name in ["dermamnist","retinamnist","breastmnist","tissuemnist","organsmnist","octmnist",\
    "pneumoniamnist"]:

        info = INFO[dset_name]
        task = info['task']
        n_channels = info['n_channels']
        num_cls = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        # preprocessing
        train_data_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        test_data_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        trainset = DataClass(split='train', transform=train_data_transform, download=True,as_rgb=True)
        valset = DataClass(split='val', transform=test_data_transform, download=True,as_rgb=True)
        testset = DataClass(split='test', transform=test_data_transform, download=True,as_rgb=True)

        #print(trainset)

        if valid:
            return trainset, valset, testset, num_cls,n_channels
        else:
            return trainset, testset, num_cls,n_channels


    elif dset_name == "cifar10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        '''cutout_size = 16
 
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            CIFAR10Policy(),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
 
        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])'''

        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        num_cls = 10

        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True,
                                               transform=cifar_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(cifar_transform)
            fullset2 = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=transform_train2)

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


    elif dset_name == "cifar100":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        '''cutout_size = 16
 
        cifar100_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            CIFAR10Policy(),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
 
        cifar100_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])'''

        cifar100_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        cifar100_tst_transform = transforms.Compose([
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

            if transform2:
                trainset2, _ = random_split(fullset2, [num_trn, num_val])
                return trainset2, valset, testset, num_cls
            else:
                return trainset, valset, testset, num_cls
        else:
            if transform2:
                return fullset, fullset2, testset, num_cls
            else:
                return fullset, testset, num_cls


    elif dset_name == "svhn":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        svhn_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        svhn_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        num_cls = 10

        fullset = torchvision.datasets.SVHN(root=datadir, split='train', download=True, transform=svhn_transform)
        testset = torchvision.datasets.SVHN(root=datadir, split='test', download=True, transform=svhn_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(svhn_transform)
            fullset2 = torchvision.datasets.SVHN(root=datadir, train=True, download=True, transform=transform_train2)

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


    elif dset_name == "kmnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        kmnist_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
        ])

        kmnist_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
        ])

        num_cls = 10

        fullset = torchvision.datasets.KMNIST(root=datadir, train=True, download=True, transform=kmnist_transform)
        testset = torchvision.datasets.KMNIST(root=datadir, train=False, download=True,
                                              transform=kmnist_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(kmnist_transform)
            fullset2 = torchvision.datasets.KMNIST(root=datadir, train=True, download=True, transform=transform_train2)

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

    elif dset_name == 'camelyon':
    
        dataset=get_dataset(dataset='camelyon17',root_dir=datadir,download=True)
        train_data= dataset.get_subset('train',frac=.9,transform=transforms.Compose([
            # you can add other transformations in this listsize([32,32])
            #transforms.CenterCrop([32,32]),
            transforms.Resize([96,96]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))

        val_data= dataset.get_subset('train',frac=.1,transform=transforms.Compose([
            # you can add other transformations in this listsize([32,32])
            #transforms.CenterCrop([32,32]),
            transforms.Resize([96,96]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))

        test_data= dataset.get_subset('test',frac=1,transform=transforms.Compose([
            # you can add other transformations in this listsize([32,32])
            #transforms.CenterCrop([32,32]),
            transforms.Resize([96,96]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))

        num_cls = 2

        if valid:
            return train_data, val_data, test_data, num_cls
        else:
            return train_data, test_data, num_cls

    elif dset_name == 'cars':

        '''jitter_params = 0.4
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jitter_params, contrast=jitter_params, saturation=jitter_params, hue=0),
            ImageNetPolicy(),
            cutout(24, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])'''
        # RandomBrightness(-0.25, 0.25),

        crop_size = 96
        transform_train = transforms.Compose([transforms.Resize(crop_size),transforms.RandomCrop(crop_size),\
        transforms.RandomRotation(45),transforms.RandomHorizontalFlip(),transforms.ToTensor(),\
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        '''transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])'''

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

    elif dset_name == 'flowers':

        crop_size = 96

        data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomRotation(45),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'valid': transforms.Compose([
                    transforms.Resize(100),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(100),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }

    
        image_datasets = {
                x: torchvision.datasets.ImageFolder(root=datadir + '/flower/' + x, transform=data_transforms[x])
                for x in list(data_transforms.keys())
            }

        
        num_cls = 102

        if valid:
            return image_datasets['train'], image_datasets['valid'], image_datasets['test'], num_cls
        else:
            return image_datasets['train'], image_datasets['test'], num_cls


    elif dset_name == "stl10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        stl10_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        stl10_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        num_cls = 10

        fullset = torchvision.datasets.STL10(root=datadir, split='train', download=True, transform=stl10_transform)
        testset = torchvision.datasets.STL10(root=datadir, split='test', download=True, transform=stl10_tst_transform)

        if transform2:
            transform_train2 = TransformTwice(stl10_transform)
            fullset2 = torchvision.datasets.STL10(root=datadir, train=True, download=True, transform=transform_train2)

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


    elif dset_name == "emnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        emnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        emnist_tst_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        num_cls = 10

        fullset = torchvision.datasets.EMNIST(root=datadir, split='digits', train=True, download=True,
                                              transform=emnist_transform)
        testset = torchvision.datasets.EMNIST(root=datadir, split='digits', train=False, download=True,
                                              transform=emnist_tst_transform)
        if transform2:
            transform_train2 = TransformTwice(emnist_transform)
            fullset2 = torchvision.datasets.EMNIST(root=datadir, train=True, download=True, transform=transform_train2)

        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(kwargs['classimb_ratio'] * num_cls),
                                                replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
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


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>    transforms.Resize(256),
        >>>    ImageNetPolicy(),
        >>>    transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>    transforms.Resize(256),
        >>>    CIFAR10Policy(),
        >>>    transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        Subranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        Subfunc = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = Subfunc[operation1]
        self.magnitude1 = Subranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = Subfunc[operation2]
        self.magnitude2 = Subranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img

def pertub(mean=0.0,sigma=1.0):

    def _pertub(sample):
        if sample.size()[1] != mean.size():
            if len(mean.size()) ==0 and len(sigma.size()) == 0:
                mean = mean*torch.mean(sample.size()[1])
                sigma = sigma*torch.eye(sample.size()[1])

            else:
                print("Dimension mismatch")
                exit(1)

        return sample+ torch.distributions.Normal(mean, sigma)


    return _pertub


def cutout(mask_size, p, cutout_inside, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


