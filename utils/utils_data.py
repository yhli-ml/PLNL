import os
import numpy as np
from copy import deepcopy
from decimal import Decimal
from math import comb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomResizedCrop, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from datasets.mnist import MNIST, FMNIST, KMNIST
from datasets.cifar import *
from datasets.svhn import *
from datasets.tinyimagenet import *
from datasets.stl import *
from datasets.mini import *

np.random.seed(2)

def class_prior(comp_labels):
    count = (torch.sum(comp_labels, dim=0)/comp_labels.size(0)).numpy()
    return count

# generate comp labels via given number
def generate_multi_comp_labels(labels, nc=1):
    k = max(labels) + 1
    n = len(labels)
    index_ins = np.arange(n) # torch type
    realY = np.zeros([n, k])
    realY[index_ins, labels] = 1
    partialY = np.ones([n, k])

    labels_hat = np.array(deepcopy(labels))
    candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0) # candidate labels without true class
    mask = np.ones((len(labels_hat), k), dtype=bool)
    for i in range(nc):
        mask[np.arange(n), labels_hat] = False
        candidates_ = candidates[mask].reshape(n, k-1-i)
        idx = np.random.randint(0, k-1-i, n)
        comp_labels = candidates_[np.arange(n), np.array(idx)]
        partialY[index_ins, torch.from_numpy(comp_labels)] = 0
        labels_hat = comp_labels
    partialY = torch.from_numpy(partialY)
    return 1-partialY # note that matrix 1-partialY whose value equals 1 where complementary label lies and 0 otherwise

# generate comp labels via uniform distribution
def generate_uniform_comp_labels(labels):
    labels = torch.tensor(labels)
    if torch.min(labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(labels) == 1:
        labels = labels - 1

    K = torch.max(labels) - torch.min(labels) + 1
    n = labels.shape[0]
    #cardinality = 2**K - 2 
    cardinality = Decimal(2)**K.item() - Decimal(2) # 2^100 overflow
    print(cardinality)
    number = torch.tensor([comb(K, i+1) for i in range(K-1)]) # 0 to K-2, convert list to tensor
    frequency_dis = number / float(cardinality)
    prob_dis = torch.zeros(K-1) # tensor of K-1
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float() # tensor: n
    mask_n = torch.ones(n) # n is the number of train_data
    partialY = torch.ones(n, K)
    temp_nc_train_labels = 0 # save temp number of comp train_labels
    
    for j in range(n): # for each instance
        if j % 10000 == 0:
            print("current index:", j)
        for jj in range(K-1): # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_nc_train_labels = jj+1 # decide the number of complementary train_labels
                mask_n[j] = 0
                break
        candidates = torch.from_numpy(np.random.permutation(K.item())) # because K is tensor type
        candidates = candidates[candidates!=labels[j]]
        temp_comp_train_labels = candidates[:temp_nc_train_labels]
        
        for kk in range(len(temp_comp_train_labels)):
            partialY[j, temp_comp_train_labels[kk]] = 0 # fulfill the partial label matrix
    return 1-partialY


def My_dataloaders(args):
    print('Preparing Data...')
    # train_loader
    if args.dataset == 'mnist':
        normalize = Normalize((0.1307,), (0.3081,))
        train_dataset = MNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'kmnist':
        normalize = Normalize((0.1307,), (0.3081,))
        train_dataset = KMNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'fmnist':
        normalize = Normalize((0.2860,), (0.3530,))
        train_dataset = FMNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'cifar10':
        normalize = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        train_dataset = MY_CIFAR10(root=args.data_dir, train=True, download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'cifar100':
        normalize = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_dataset = MY_CIFAR100(root=args.data_dir, train=True, download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'svhn':
        normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_dataset = MY_SVHN(root=args.data_dir, split='train', download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'tinyimagenet':
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = My_TinyImageNet(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'mini-imagenet':
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = MINI(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'stl10':
        normalize = Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2235))
        train_dataset = MY_STL10(root=args.data_dir, split='train', distr=args.distr, nc=args.nc)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
    train_loader_ = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
    # test_loader
    if args.dataset in ['tinyimagenet', 'mini-imagenet']:
        test_transform = Compose([
            Resize(224),
            RandomResizedCrop(224),
            ToTensor(),
            normalize,
        ])
    else:
        test_transform = Compose([
            ToTensor(),
            normalize,
        ])
    if args.dataset == 'mnist':
        test_dataset = MNIST(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'kmnist':
        test_dataset = KMNIST(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'fmnist':
        test_dataset = FMNIST(root=args.data_dir, train=False, transform=test_transform)
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'svhn':
        test_dataset = datasets.SVHN(root=args.data_dir, split='test', transform=test_transform, download=True)
    elif args.dataset == 'tinyimagenet':
        test_dataset = My_TinyImageNet(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'mini-imagenet':
        test_dataset = MINI(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'stl10':
        test_dataset = datasets.STL10(root=args.data_dir, split='test', transform=test_transform, download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)
    
    print('Dataset {0} loaded already\n'
          'num_classes: {2}\n'
          'Training Set -- Num_samples: {1}\n'
          'Testing  Set -- Num_samples: {3}'.format(args.dataset, train_dataset.n, train_dataset.k, len(test_dataset)))
    return train_loader,train_loader_, test_loader

def My_ga_nn_dataloaders(args):
    print('Preparing Data...')
    # train_loader
    if args.dataset == 'mnist':
        normalize = Normalize((0.1307,), (0.3081,))
        train_dataset = MNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'kmnist':
        normalize = Normalize((0.1307,), (0.3081,))
        train_dataset = KMNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'fmnist':
        normalize = Normalize((0.2860,), (0.3530,))
        train_dataset = FMNIST(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'cifar10':
        normalize = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        train_dataset = MY_CIFAR10(root=args.data_dir, train=True, download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'cifar100':
        normalize = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_dataset = MY_CIFAR100(root=args.data_dir, train=True, download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'svhn':
        normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_dataset = MY_SVHN(root=args.data_dir, split='train', download=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'tinyimagenet':
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = My_TinyImageNet(root=args.data_dir, train=True, distr=args.distr, nc=args.nc)
    elif args.dataset == 'stl':
        normalize = Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2235))
        train_dataset = MY_STL10(root=args.data_dir, split='train', distr=args.distr, nc=args.nc)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
    # test_loader
    if args.dataset == 'tinyimagenet':
        test_transform = Compose([
            Resize(224),
            RandomResizedCrop(224),
            ToTensor(),
            normalize,
        ])
    else:
        test_transform = Compose([
            ToTensor(),
            normalize,
        ])
    class_prior_ = class_prior(train_dataset.comp_labels)

    if args.dataset == 'mnist':
        test_dataset = MNIST(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'kmnist':
        test_dataset = KMNIST(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'fmnist':
        test_dataset = FMNIST(root=args.data_dir, train=False, transform=test_transform)
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'svhn':
        test_dataset = datasets.SVHN(root=args.data_dir, split='test', transform=test_transform, download=True)
    elif args.dataset == 'tinyimagenet':
        test_dataset = My_TinyImageNet(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'stl':
        test_dataset = datasets.STL10(root=args.data_dir, split='test', transform=test_transform, download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)
    
    print('Dataset {0} loaded already\n'
          'num_classes: {2}\n'
          'Training Set -- Num_samples: {1}\n'
          'Testing  Set -- Num_samples: {3}'.format(args.dataset, train_dataset.n, train_dataset.k, len(test_dataset)))
    return train_loader, test_loader, class_prior_