import os
from abc import abstractmethod
from typing import List
import time

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

    def __init__(self, data_dir, max_length: int = 256, num_labels: int = 30, valid: bool = False, test: bool = False, **kwargs):
        super(BaselineDataset, self).__init__()

        self.data_dir = data_dir
        self.max_length = max_length
        self.num_labels = num_labels

        self.data = None
        if not valid and not test:
            self.data = pd.read_csv(os.path.join(
                self.data_dir, BaselineDataset.train_file_name), encoding='utf-8')
        elif valid:
            self.data = pd.read_csv(os.path.join(
                self.data_dir, BaselineDataset.valid_file_name), encoding='utf-8')
        elif test:
            self.data = pd.read_csv(os.path.join(
                self.data_dir, BaselineDataset.test_file_name), encoding='utf-8')

        additionals = kwargs.get("additional", None)
        if additionals is not None:
            print(additionals)
            additional_df = [self.data]
            for file_name in additionals:
                additional_df.append(pd.read_csv(os.path.join(self.data_dir, file_name), encoding='utf-8'))
            self.data = pd.concat(additional_df)
        
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
    
    def get_special_token_num(self) -> int: return 0

    def get_num_unique_ids(self) -> List[int]: return list(set(self.data['id']))

    def save_preprocessed_data(self, save_file_name = "preprocessed_data.csv"):
        self.data.to_csv(os.path.join(self.data_dir, save_file_name))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_preprocessor(self, preprocesor: Preprocessor):
        self.preprocessor = preprocesor

    def set_augmentation(self, augmentation: Augmentation):
        self.augmentation = augmentation

    def save_data(self, save_file: str):
        self.data.to_csv(save_file)

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

        if not hasattr(self.data, 't5_sbj_ts_idx'):
            self.data['t5_sbj_ts_idx'] = -1
            self.data['t5_sbj_te_idx'] = -1
            self.data['t5_obj_ts_idx'] = -1
            self.data['t5_obj_te_idx'] = -1

        sentence = self.data['t5_inputs'].iloc[index]
        label    = self.data['label'].iloc[index]

        tokenized_sentence = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        t5_sbj_ts_idx = self.data['t5_sbj_ts_idx'].iloc[index]
        t5_sbj_te_idx = self.data['t5_sbj_te_idx'].iloc[index]
        t5_obj_ts_idx = self.data['t5_obj_ts_idx'].iloc[index]
        t5_obj_te_idx = self.data['t5_obj_te_idx'].iloc[index]

        if t5_sbj_ts_idx == -1:
            # hasn't been calculated yet...

            t5_sbj_s_idx = self.data['t5_sbj_s_idx'].iloc[index]
            t5_sbj_e_idx = self.data['t5_sbj_e_idx'].iloc[index]
            t5_obj_s_idx = self.data['t5_obj_s_idx'].iloc[index]
            t5_obj_e_idx = self.data['t5_obj_e_idx'].iloc[index]

            temp = self.tokenizer(
                sentence[:t5_sbj_s_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            t5_sbj_ts_idx = temp['input_ids'].size(1)

            temp = self.tokenizer(
                sentence[:t5_sbj_e_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            t5_sbj_te_idx = temp['input_ids'].size(1)

            temp = self.tokenizer(
                sentence[:t5_obj_s_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            t5_obj_ts_idx = temp['input_ids'].size(1)

            temp = self.tokenizer(
                sentence[:t5_obj_e_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            t5_obj_te_idx = temp['input_ids'].size(1)

            self.data.loc[self.data.index[index], 't5_sbj_ts_idx'] = t5_sbj_ts_idx
            self.data.loc[self.data.index[index], 't5_sbj_te_idx'] = t5_sbj_te_idx
            self.data.loc[self.data.index[index], 't5_obj_ts_idx'] = t5_obj_ts_idx
            self.data.loc[self.data.index[index], 't5_obj_te_idx'] = t5_obj_te_idx
            # it looks weird 
            # since self.data['column'].iloc[idx] == self.__getitem__('column').__setitem__(idx, item)
            # therefore it is problematic when assigning with double indexing

            # print("Generate index:")
            # print(t5_sbj_ts_idx, t5_sbj_te_idx, t5_obj_ts_idx, t5_obj_te_idx)

        out = {key: value[0] for key, value in tokenized_sentence.items()}
        out['labels'] = torch.tensor(label)
        out['entity_token_idx'] = torch.tensor([[t5_sbj_ts_idx, t5_sbj_te_idx], [t5_obj_ts_idx, t5_obj_te_idx]])

        # dict of {'input_ids', 'attention_mask', 'labels', 'entity_token_idx'}
        return out