import torch
from torch.utils.data import Dataset
from typing import Tuple
from torch.utils.data import DataLoader, random_split, SequentialSampler
from torchvision import transforms


class StandardDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super(StandardDataset, self).__init__()
        self.x = x
        self.y = y
        self.len = len(y)

    def __len__(self):
        return self.len

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


class ImageClsDataset:
    def __init__(self,
                 dataset_class,
                 save_dir: str,
                 train_trans: transforms.Compose,
                 val_trans: transforms.Compose,
                 test_trans: transforms.Compose,
                 n_train_val_obs: int = -1,
                 n_test_obs: int = -1,
                 val_size: float = 0.2):

        train_val_dataset = dataset_class(root=save_dir, train=True, download=True, transform=None)
        if n_train_val_obs > - 1:
            train_val_dataset.data = train_val_dataset.data[:n_train_val_obs]
            train_val_dataset.targets = train_val_dataset.targets[:n_train_val_obs]
        train_dataset, val_dataset = self.train_val_split(train_val_dataset, val_size=val_size)
        train_dataset.dataset.transform = train_trans
        val_dataset.dataset.transform = val_trans

        test_dataset = dataset_class(root=save_dir, train=False, download=True, transform=None)
        if n_test_obs > - 1:
            test_dataset.data = test_dataset.data[:n_test_obs]
            test_dataset.targets = test_dataset.targets[:n_test_obs]
        test_dataset.transform = test_trans

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def train_val_split(train_val_dataset, val_size: float):
        train_len = int(len(train_val_dataset) * (1 - val_size))
        val_len = int(len(train_val_dataset) * val_size)
        return random_split(dataset=train_val_dataset, lengths=[train_len, val_len])

    def create_loaders(self, batch_size_train: int, batch_size_val: int, batch_size_test: int, sequential_training_loader: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        :param batch_size_train: size of the train batches
        :type batch_size_train: int

        :param batch_size_val: size of the val batches
        :type batch_size_val: int

        :param batch_size_test: size of the test batches
        :type batch_size_test: int

        :param sequential_training_loader: determines if the training loader samples sequentially
        :type sequential_training_loader: bool

        :return: Tuple[DataLoader] train_loader(random or sequential), val_loader(sequential), test_loader(sequential)
        """
        val_sampler = SequentialSampler(data_source=self.val_dataset)
        if sequential_training_loader:
            train_sampler = SequentialSampler(data_source=self.train_dataset)
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size_train, sampler=train_sampler)
        else:
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size_train)
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size_val, sampler=val_sampler)

        test_sampler = SequentialSampler(data_source=self.test_dataset)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size_test, sampler=test_sampler)
        return train_loader, val_loader, test_loader
