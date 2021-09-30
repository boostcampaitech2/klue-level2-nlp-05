import os
from abc import abstractmethod

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..preprocessor.preprocessors import Preprocessor
from ..augmentation.augmentations import Augmentation


class BaselineDataset(Dataset):
    train_file_name = "train.csv"
    valid_file_name = "valid.csv"
    test_file_name  = "test_data.csv"

    def __init__(self, data_dir, max_seq_len: int = 256, num_labels: int = 30):
        super(BaselineDataset, self).__init__()

        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels

        self.data = pd.read_csv(os.path.join(
            self.data_dir, BaselineDataset.train_file_name), encoding='utf-8')

        self.tokenizer = None
        self.preprocessor = None
        self.augmentation = None

    def __getitem__(self, index):
        if self.tokenizer is None:
            raise AttributeError(
                "please first set tokenizer with self.set_tokenizer() method")

        sentence = self.data['sentence'].iloc[index]
        concat_entity = self.data['concat_entity'].iloc[index]

        if self.augmentation is not None:
            sentence = self.augmentation(sentence)
            # augmentation processes directly on the string input

        sentence_tokens = self.tokenizer.tokenize(sentence)
        entity_tokens   = self.tokenizer.tokenize(concat_entity)

        sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        entity_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)

        return {'sentence_ids': sentence_ids,
                'entity_ids': entity_ids
                }

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        self.data = self.preprocessor(self.data)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_preprocessor(self, preprocesor: Preprocessor):
        self.preprocessor = preprocesor

    def set_augmentation(self, augmentation: Augmentation):
        self.augmentation = augmentation

    @staticmethod
    def set_train_file(file_name):
        BaselineDataset.train_file_name = file_name

    @staticmethod
    def set_valid_file(file_name):
        BaselineDataset.valid_file_name = file_name

    @staticmethod
    def set_test_file(file_name):
        BaselineDataset.test_file_name = file_name
