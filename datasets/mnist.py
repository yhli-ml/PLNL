import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision import models, utils, datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, Resize, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage
import numpy as np
import gzip
import os
from PIL import Image
import copy 
from decimal import Decimal
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy


class MNIST(VisionDataset):
    base_folder = 'mnist'
    normalize = Normalize((0.1307,), (0.3081,))
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    
    def __init__(self, root, train=True, transform=None, transform1=None, distr=0, nc=1):
        self.train = train
        self.root = root
        self.distr = distr
        self.nc = nc
        self.images = []
        self.targets = []

        self._make_dataset(self.train)

        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()

        self.n = len(self.targets)
        self.k = max(self.targets) + 1
        self.confidence = torch.zeros(self.n) # note that this only means confidence(n, 1) not masks(n, k)
        self.confident_true_labels = torch.zeros(self.n, self.k)
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes=self.k)

        if transform == None:
            self.transform=Compose([
                ToTensor(),
                self.normalize,
            ])

            self.transform1=Compose([
                RandomHorizontalFlip(),
                RandomCrop(28, 4, padding_mode='reflect'),
                ToTensor(),
                Cutout(n_holes=1, length=16),
                self.normalize,
            ])
        else:
            self.transform = transform
            self.transform1 = transform

    def _make_dataset(self, train=True):
        paths = []
        for fname in self.files:
            paths.append(os.path.join(self.root, self.base_folder, fname))

        with gzip.open(paths[0], 'rb') as lbpath:
            train_target = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[1], 'rb') as imgpath:
            train_img = np.frombuffer(imgpath.read(), np.uint8, offset=16)
            train_img = train_img.reshape(-1, 28, 28)

        with gzip.open(paths[2], 'rb') as lbpath:
            test_target = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[3], 'rb') as imgpath:
            test_img = np.frombuffer(imgpath.read(), np.uint8, offset=16)
            test_img = test_img.reshape(-1, 28, 28)

        if train:
            self.images = train_img
            self.targets = train_target.tolist()
        else:
            self.images = test_img
            self.targets = test_target.tolist()

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):
        each_img_ori, each_comp_label, each_target, each_confidence, each_confident_true_label = self.images[index], self.comp_labels[index], self.targets[index], self.confidence[index], self.confident_true_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        each_img_ori = Image.fromarray(each_img_ori)
        if self.transform is not None:
            each_img = self.transform(each_img_ori)
            each_img_aug1 = self.transform1(each_img_ori)
            each_img_aug2 = self.transform1(each_img_ori)
        if self.train:
            return each_img, each_img_aug1, each_img_aug2, each_comp_label, each_confidence, each_confident_true_label, index
        else:
            return each_img, each_target

    # generate comp labels via given number(returns torch tensor)
    def generate_multi_comp_labels(self):
        k = max(self.targets) + 1
        n = len(self.targets)
        index_ins = np.arange(n) # torch type
        realY = np.zeros([n, k])
        realY[index_ins, self.targets] = 1
        partialY = np.ones([n, k])

        labels_hat = np.array(copy.deepcopy(self.targets))
        candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0) # candidate labels without true class
        mask = np.ones((len(labels_hat), k), dtype=bool)
        for i in range(self.nc):
            mask[np.arange(n), labels_hat] = False
            candidates_ = candidates[mask].reshape(n, k-1-i)
            idx = np.random.randint(0, k-1-i, n)
            comp_labels = candidates_[np.arange(n), np.array(idx)]
            partialY[index_ins, torch.from_numpy(comp_labels)] = 0
            labels_hat = comp_labels
        partialY = torch.from_numpy(partialY)
        return 1-partialY # note that matrix 1-partialY whose value equals 1 where complementary label lies and 0 otherwise
    
    # generate comp labels via uniform distribution(returns torch tensor) 
    def generate_uniform_comp_labels(self):
        labels = torch.tensor(self.targets)
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
    

class KMNIST(MNIST):
    base_folder = 'kmnist'
    normalize = Normalize((0.1307,), (0.3081,))


class FMNIST(MNIST):
    base_folder = 'fmnist'
    normalize = Normalize((0.2860,), (0.3530,))