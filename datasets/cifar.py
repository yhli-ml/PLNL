from PIL import Image
import os
import os.path
from decimal import Decimal
import numpy as np
import pickle
from typing import Any, Optional, Callable, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage
import copy
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from augment.randaugment import RandomAugment

class MY_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    normalize = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    def __init__(
            self, 
            root: str, 
            train=True, 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None,
            download=False, 
            distr=0, nc=1,
    ) -> None:
        super(MY_CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.\nYou can use download=True to download it')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to Height-Width-Channel
        
        # number of complementary labels
        self.nc = nc
        # confidence to generate masks for sample selection
        self.k = max(self.targets) + 1
        self.n = len(self.targets)
        self.confidence = torch.zeros(self.n) # note that this only means confidence(n, 1) not masks(n, k)
        self.confident_true_labels = torch.zeros(self.n, self.k)
        # generate complementary labels
        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()
        # One-hot version of true labels
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes=self.k)
        ## data augmentation
        # weak augmentation
        self.transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            self.normalize,
        ])
        # strong augmentation
        self.transform1=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            CIFAR10Policy(),
            ToTensor(),
            self.normalize,
        ])
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        each_img_ori, each_comp_label, each_target, each_confidence, each_confident_true_label = self.data[index], self.comp_labels[index], self.targets[index], self.confidence[index], self.confident_true_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        each_img_ori = Image.fromarray(each_img_ori)
        if self.transform is not None:
            each_img_w = self.transform(each_img_ori)
            each_img_w_2 = self.transform(each_img_ori)
            each_img_s = self.transform1(each_img_ori)
            # each_img_aug2 = self.transform1(each_img_ori)
        if self.target_transform is not None:
            each_target = self.target_transform(each_target)
        return each_img_w, each_img_w_2, each_img_s, each_comp_label,each_confidence, each_confident_true_label, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    # generate comp labels via given number(returns torch tensor) one-hot version
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
    
    # generate comp labels via uniform distribution(returns torch tensor) one-hot version
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
            # if j % 10000 == 0:
            #     print("current index:", j)
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

class MY_CIFAR100(MY_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
        This is a subclass of 'MY_CIFAR10' dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    normalize = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    def __init__(
                self, 
                root: str, 
                train=True, 
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None,
                download=False, 
                distr=0, nc=1,
        ) -> None:
        super(MY_CIFAR100, self).__init__(root, train, transform, target_transform, download, distr, nc)
        self.transform1=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            RandomAugment(3, 5), # strong augmentation
            ToTensor(),
            self.normalize, 
        ])


class MY_CIFAR100_POCR(MY_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
        This is a subclass of 'MY_CIFAR10' dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    normalize = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    def __init__(
                self, 
                root: str, 
                train=True, 
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None,
                download=False, 
                distr=0, nc=1,
        ) -> None:
        super(MY_CIFAR100_POCR, self).__init__(root, train, transform, target_transform, download, distr, nc)
        self.transform1=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            self.normalize, 
        ])
    
# # CIFAR10 used for ViT in blip2
# class MY_CIFAR10_transformer(VisionDataset):
#     """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.

#     """
#     base_folder = 'cifar-10-batches-py'
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = "cifar-10-python.tar.gz"
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#     meta = {
#         'filename': 'batches.meta',
#         'key': 'label_names',
#         'md5': '5ff9c542aee3614f3951f8cda6e48888',
#     }
#     normalize = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#     def __init__(
#             self, 
#             root: str, 
#             train=True, 
#             transform: Optional[Callable] = None, 
#             target_transform: Optional[Callable] = None,
#             download=False, 
#             distr=0, nc=1,
#     ) -> None:
#         super(MY_CIFAR10_transformer, self).__init__(root, transform=transform,
#                                       target_transform=target_transform)
#         self.train = train  # training set or test set
#         if download:
#             self.download()
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.\nYou can use download=True to download it')
#         if self.train:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list
#         self.data = []
#         self.targets = []
#         # now load the picked numpy arrays
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 entry = pickle.load(f, encoding='latin1')
#                 self.data.append(entry['data'])
#                 if 'labels' in entry:
#                     self.targets.extend(entry['labels'])
#                 else:
#                     self.targets.extend(entry['fine_labels'])
#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to Height-Width-Channel
        
#         # number of complementary labels
#         self.nc = nc
#         # confidence to generate masks for sample selection
#         self.k = max(self.targets) + 1
#         self.n = len(self.targets)
#         # generate complementary labels
#         if distr==0:
#             self.comp_labels = self.generate_multi_comp_labels()
#         else:
#             self.comp_labels = self.generate_uniform_comp_labels()
#         # One-hot version of true labels
#         self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes=self.k)
#         # transform for ViT
#         self.transform=transform

#         self._load_meta()

#     def _load_meta(self):
#         path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#         if not check_integrity(path, self.meta['md5']):
#             raise RuntimeError('Dataset metadata file not found or corrupted.' +
#                                ' You can use download=True to download it')
#         with open(path, 'rb') as infile:
#             data = pickle.load(infile, encoding='latin1')
#             self.classes = data[self.meta['key']]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         each_img_ori = self.data[index]
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         each_img_ori = Image.fromarray(each_img_ori)
#         if self.transform is not None:
#             each_img = self.transform(each_img_ori)
#         return each_img, index

#     def __len__(self) -> int:
#         return len(self.data)

#     def _check_integrity(self) -> bool:
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True

#     def download(self) -> None:
#         if self._check_integrity():
#             return
#         download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

#     def extra_repr(self) -> str:
#         return "Split: {}".format("Train" if self.train is True else "Test")

#     # generate comp labels via given number(returns torch tensor) one-hot version
#     def generate_multi_comp_labels(self):
#         k = max(self.targets) + 1
#         n = len(self.targets)
#         index_ins = np.arange(n) # torch type
#         realY = np.zeros([n, k])
#         realY[index_ins, self.targets] = 1
#         partialY = np.ones([n, k])
#         labels_hat = np.array(copy.deepcopy(self.targets))
#         candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0) # candidate labels without true class
#         mask = np.ones((len(labels_hat), k), dtype=bool)
#         for i in range(self.nc):
#             mask[np.arange(n), labels_hat] = False
#             candidates_ = candidates[mask].reshape(n, k-1-i)
#             idx = np.random.randint(0, k-1-i, n)
#             comp_labels = candidates_[np.arange(n), np.array(idx)]
#             partialY[index_ins, torch.from_numpy(comp_labels)] = 0
#             labels_hat = comp_labels
#         partialY = torch.from_numpy(partialY)
#         return 1-partialY # note that matrix 1-partialY whose value equals 1 where complementary label lies and 0 otherwise
    
#     # generate comp labels via uniform distribution(returns torch tensor) one-hot version
#     def generate_uniform_comp_labels(self):
#         labels = torch.tensor(self.targets)
#         if torch.min(labels) > 1:
#             raise RuntimeError('testError')
#         elif torch.min(labels) == 1:
#             labels = labels - 1
#         K = torch.max(labels) - torch.min(labels) + 1
#         n = labels.shape[0]
#         #cardinality = 2**K - 2  
#         cardinality = Decimal(2)**K.item() - Decimal(2) # 2^100 overflow
#         number = torch.tensor([comb(K, i+1) for i in range(K-1)]) # 0 to K-2, convert list to tensor
#         frequency_dis = number / float(cardinality)
#         prob_dis = torch.zeros(K-1) # tensor of K-1
#         for i in range(K-1):
#             if i == 0:
#                 prob_dis[i] = frequency_dis[i]
#             else:
#                 prob_dis[i] = frequency_dis[i]+prob_dis[i-1]
#         random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float() # tensor: n
#         mask_n = torch.ones(n) # n is the number of train_data
#         partialY = torch.ones(n, K)
#         temp_nc_train_labels = 0 # save temp number of comp train_labels
#         for j in range(n): # for each instance
#             # if j % 10000 == 0:
#             #     print("current index:", j)
#             for jj in range(K-1): # 0 to K-2
#                 if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
#                     temp_nc_train_labels = jj+1 # decide the number of complementary train_labels
#                     mask_n[j] = 0
#                     break
#             candidates = torch.from_numpy(np.random.permutation(K.item())) # because K is tensor type
#             candidates = candidates[candidates!=labels[j]]
#             temp_comp_train_labels = candidates[:temp_nc_train_labels]
#             for kk in range(len(temp_comp_train_labels)):
#                 partialY[j, temp_comp_train_labels[kk]] = 0 # fulfill the partial label matrix
#         return 1-partialY
    
# class MY_CIFAR100_transformer(MY_CIFAR10_transformer):
#     """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#         This is a subclass of 'MY_CIFAR10' dataset.
#     """
#     base_folder = 'cifar-100-python'
#     url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     filename = "cifar-100-python.tar.gz"
#     tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
#     train_list = [
#         ['train', '16019d7e3df5f24257cddd939b257f8d'],
#     ]
#     test_list = [
#         ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
#     ]
#     meta = {
#         'filename': 'meta',
#         'key': 'fine_label_names',
#         'md5': '7973b15100ade9c7d40fb424638fde48',
#     }
#     normalize = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#     def __init__(
#                 self, 
#                 root: str, 
#                 train=True, 
#                 transform: Optional[Callable] = None, 
#                 target_transform: Optional[Callable] = None,
#                 download=False, 
#                 distr=0, nc=1,
#         ) -> None:
#         super(MY_CIFAR100_transformer, self).__init__(root, train, transform, target_transform, download, distr, nc)