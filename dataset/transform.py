from typing import List, Set, Dict, Tuple

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

class BaseTransform(object):
    def __init__(self, output_size: List[int]) -> None:
        self.output_size = output_size
        self.transform = T.Resize(self.output_size)

    def __call__(self, inputs):
        return self.transform(inputs)