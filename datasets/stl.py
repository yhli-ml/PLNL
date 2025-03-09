from PIL import Image
import os
import os.path
from decimal import Decimal
import numpy as np
import pickle
from typing import Any, Optional, cast, Callable, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage
import copy
from scipy.special import comb
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from augment.randaugment import RandomAugment


class MY_STL10(VisionDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
            standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    normalize = Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2235))
    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        distr=0, nc=1,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            self.labels = self.labels.tolist()

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        # number of complementary labels
        self.nc = nc
        # confidence to generate masks for sample selection
        self.k = max(self.labels) + 1
        self.n = len(self.labels)
        self.confidence = torch.zeros(self.n) # note that this only means confidence(n, 1) not masks(n, k)
        self.confident_true_labels = torch.zeros(self.n, self.k)
        # generate complementary labels
        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()
        # One-hot version of true labels
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.labels), num_classes=self.k)
        # data augmentation
        # weak augmentation
        self.transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(96, 12, padding_mode='reflect'),
            ToTensor(),
            self.normalize,
        ])
        # strong augmentation
        self.transform1=Compose([
            RandomHorizontalFlip(),
            RandomCrop(96, 12, padding_mode='reflect'),
            RandomAugment(3, 5),
            ToTensor(),
            self.normalize,
        ])

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        each_img_ori, each_comp_label, each_target, each_confidence, each_confident_true_label = self.data[index], self.comp_labels[index], self.labels[index], self.confidence[index], self.confident_true_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        each_img_ori = Image.fromarray(np.transpose(each_img_ori, (1, 2, 0)))

        if self.transform is not None:
            each_img_w = self.transform(each_img_ori)
            each_img_w_2 = self.transform(each_img_ori)
            each_img_s = self.transform1(each_img_ori)
            # each_img_aug2 = self.transform1(each_img_ori)

        if self.target_transform is not None:
            each_target = self.target_transform(each_target)

        return each_img_w, each_img_w_2, each_img_s, each_comp_label, each_confidence, each_confident_true_label, index

    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]

    # generate comp labels via given number(returns torch tensor) one-hot version
    def generate_multi_comp_labels(self):
        k = max(self.labels) + 1
        n = len(self.labels)
        index_ins = np.arange(n) # torch type
        realY = np.zeros([n, k])
        realY[index_ins, self.labels] = 1
        partialY = np.ones([n, k])
        labels_hat = np.array(copy.deepcopy(self.labels))
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
        labels = torch.tensor(self.labels)
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
    
class MY_STL10_transformer(VisionDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
            standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    normalize = Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2235))
    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        distr=0, nc=1,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            self.labels = self.labels.tolist()

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        # number of complementary labels
        self.nc = nc
        # confidence to generate masks for sample selection
        self.k = max(self.labels) + 1
        self.n = len(self.labels)
        # generate complementary labels
        if distr==0:
            self.comp_labels = self.generate_multi_comp_labels()
        else:
            self.comp_labels = self.generate_uniform_comp_labels()
        # One-hot version of true labels
        self.true_labels = torch.nn.functional.one_hot(torch.tensor(self.labels), num_classes=self.k)

        self.transform=transform

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

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
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]

    # generate comp labels via given number(returns torch tensor) one-hot version
    def generate_multi_comp_labels(self):
        k = max(self.labels) + 1
        n = len(self.labels)
        index_ins = np.arange(n) # torch type
        realY = np.zeros([n, k])
        realY[index_ins, self.labels] = 1
        partialY = np.ones([n, k])
        labels_hat = np.array(copy.deepcopy(self.labels))
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
        labels = torch.tensor(self.labels)
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