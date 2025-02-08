from PIL import Image
import os
import os.path
from decimal import Decimal
import numpy as np
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage
import copy
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import SVHNPolicy

class MY_SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, 
            distr=0, nc=1,
    ) -> None:
        super(MY_SVHN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.\nYou can use download=True to download it')
        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int32).squeeze()
        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.targets = self.labels.tolist()
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
        # data augmentation
        self.transform=Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform1=Compose([
            ToTensor(),
            Cutout(n_holes=1, length=20),
            ToPILImage(),
            SVHNPolicy(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
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
        each_img_ori = Image.fromarray(np.transpose(each_img_ori, (1, 2, 0)))
        if self.transform is not None:
            each_img = self.transform(each_img_ori)
            each_img_aug1 = self.transform1(each_img_ori)
            each_img_aug2 = self.transform1(each_img_ori)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return each_img, each_img_aug1, each_img_aug2, each_comp_label, each_confidence, each_confident_true_label, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

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
    
    def generate_multi_compl_labels(self):
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
        return 1-partialY
    
class MY_SVHN_transformer(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, 
            distr=0, nc=1,
    ) -> None:
        super(MY_SVHN_transformer, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.\nYou can use download=True to download it')
        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int32).squeeze()
        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.targets = self.labels.tolist()
        # number of complementary labels
        self.nc = nc
        # confidence to generate masks for sample selection
        self.k = max(self.targets) + 1
        self.n = len(self.targets)
        # generate complementary labels
        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()
        # One-hot version of true labels
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes=self.k)

        self.transform=transform
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        each_img_ori = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        each_img_ori = Image.fromarray(np.transpose(each_img_ori, (1, 2, 0)))
        if self.transform is not None:
            each_img = self.transform(each_img_ori)
        return each_img, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

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
    
    def generate_multi_compl_labels(self):
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
        return 1-partialY