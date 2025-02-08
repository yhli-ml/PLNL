import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision import models, utils, datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, Resize, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage
import numpy as np
import sys
import os
from PIL import Image
import copy 
from decimal import Decimal
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy
from augment.randaugment import RandomAugment


class MINI(VisionDataset):
    base_folder = 'mini-imagenet'
    normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    def __init__(self, root, train=True, transform=None, transform1=None, target_transform=None, distr=0, nc=1):
        self.train = train
        self.root_dir = root
        self.distr = distr
        self.nc = nc
        self.images = []
        self.targets = []
        self.train_dir = os.path.join(self.root_dir, self.base_folder, "train")
        self.val_dir = os.path.join(self.root_dir, self.base_folder, "val")

        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.train)

        words_file = os.path.join(self.root_dir, self.base_folder, "words.txt")
        wnids_file = os.path.join(self.root_dir, self.base_folder, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        
        self.n = len(self.targets)
        self.k = max(self.targets) + 1
        self.confidence = torch.zeros(self.n)
        self.confident_true_labels = torch.zeros(self.n, self.k)
        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes=self.k)
        
        if transform == None:
            self.transform=Compose([
                Resize(224),
                RandomHorizontalFlip(),
                RandomCrop(224, 28, padding_mode='reflect'),
                ToTensor(),
                self.normalize,
            ])
            self.transform1=Compose([
                Resize(224),
                RandomHorizontalFlip(),
                RandomCrop(224, 28, padding_mode='reflect'),
                RandomAugment(3, 5),
                ToTensor(),
                self.normalize,
            ])
        else:
            self.transform = transform
            self.transform1 = transform
        self.target_transform = target_transform

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1
        n = num_images
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])
        num_samples = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, train=True):
        if train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue
            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item[0])
                        self.targets.append(item[1])

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img_path = self.images[index]
        with open(img_path, 'rb') as f:
            img = Image.open(img_path)
            each_img_ori = img.convert('RGB')
            each_comp_label, each_target, each_confidence, each_confident_true_label = self.comp_labels[index], self.targets[index], self.confidence[index], self.confident_true_labels[index]
        if self.transform is not None:
            each_img = self.transform(each_img_ori)
            each_img_aug1 = self.transform1(each_img_ori)
            each_img_aug2 = self.transform1(each_img_ori)
        if self.target_transform is not None:
            each_target = self.target_transform(each_target)
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