import torch
from torch.utils.data import Dataset


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
