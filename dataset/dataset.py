import os
from abc import abstractmethod

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.std import TqdmDefaultWriteLock

from .preprocessor.preprocessors import Preprocessor
from .augmentation.augmentations import Augmentation


class BaselineDataset(Dataset):
    train_file_name = "train/train.csv"
    valid_file_name = "train/valid.csv"
    test_file_name  = "test/test_data.csv"

    def __init__(self, data_dir, max_length: int = 256, num_labels: int = 30, **kwargs):
        super(BaselineDataset, self).__init__()

        self.data_dir = data_dir
        self.max_length = max_length
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

        # # tokenizer.tokenize() -> List[str]
        # sentence_tokens = self.tokenizer.tokenize(sentence)
        # entity_tokens   = self.tokenizer.tokenize(concat_entity)

        # sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        # entity_ids   = self.tokenizer.convert_tokens_to_ids(entity_tokens)

        tokenized_sentence = self.tokenizer(
            concat_entity,
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        out = {key: value[0] for key, value in tokenized_sentence.items()}
        out['label'] = torch.tensor(self.data['label'].iloc[index])

        # dict of {'input_ids', 'token_type_ids', 'attention_mask', 'labels'}
        return out

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        self.data = self.preprocessor(self.data)
    
    def get_special_token_num(self): return 0

    def save_preprocessed_data(self, save_file_name = "preprocessed_data.csv"):
        self.data.to_csv(os.path.join(self.data_dir, save_file_name))

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


# class ExtendedDataset(BaselineDataset):
class EntitySpecialTokenDataset(BaselineDataset):

    def __getitem__(self, index):
        if self.tokenizer is None:
            raise AttributeError(
                "please first set tokenizer with self.set_tokenizer() method")

        sentence = self.data['sentence'].iloc[index]
        concat_entity = self.data['concat_entity'].iloc[index]

        if self.augmentation is not None:
            sentence = self.augmentation(sentence)
            
        # add special tokens 
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})

        tokenized_sentence = self.tokenizer(
            concat_entity,
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        out = {key: value[0] for key, value in tokenized_sentence.items()}
        out['label'] = torch.tensor(self.data['label'].iloc[index])
        #out['entity_ids'] = [0,1,0,0,1,0]
        #out['input_embeds'] = 
        # dict of {'input_ids', 'token_type_ids', 'attention_mask', 'labels'}
        return out

    def preprocess(self):
        self.data, self.special_tokens = self.preprocessor(self.data)
    
    def get_special_token_num(self):
        return len(self.special_tokens)


class T5Dataset(BaselineDataset):

    def __getitem__(self, index):
        if self.tokenizer is None:
            raise AttributeError(
                "please first set tokenizer with self.set_tokenizer() method")

        sentence = self.data['sentence'].iloc[index]
        label    = self.data['label'].iloc[index]

        if self.augmentation is not None:
            sentence = self.augmentation(sentence)
            # augmentation processes directly on the string input

        # # tokenizer.tokenize() -> List[str]
        # sentence_tokens = self.tokenizer.tokenize(sentence)
        # entity_tokens   = self.tokenizer.tokenize(concat_entity)

        # sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        # entity_ids   = self.tokenizer.convert_tokens_to_ids(entity_tokens)

        tokenized_sentence = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        out = {key: value[0] for key, value in tokenized_sentence.items()}
        out['labels'] = torch.tensor(label)

        # dict of {'input_ids', 'attention_mask', 'labels'}
        return out