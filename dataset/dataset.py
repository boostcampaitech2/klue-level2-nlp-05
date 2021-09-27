import torch
import torch.nn as nn
from torch.utils import data

from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self, data_dir, num_classes = 1) -> None:
        super(BaseDataset, self).__init__()
        
        self.data_dir = data_dir
        self.num_classes = num_classes

    def split_dataset(self, val_ratio: float = 0.2):
        pass