from abc import abstractmethod
from typing import Optional, Union, List

import numpy as np


class Augmentation:
    @abstractmethod
    def __call__(self, input: str) -> str:
        pass


class SimpleRandomUNK(Augmentation):

    def __init__(self,
                 unk_token: str,
                 unk_ratio: float = 0.15):

        self.unk_ratio = unk_ratio
        self.unk_token = unk_token

    def __call__(self, input: str) -> str:
        """Place `<UNK>` token at randomly selected words with the probability `unk_ratio`.

        Args:
        * input: str (input string)

        Returns:
        * processed_string: str
        """
        word_list = input.split()
        mask = np.random.rand(len(word_list)) < self.unk_ratio
        new_list = [self.unk_token if m else word for word, m
                    in zip(word_list, mask)]

        return " ".join(new_list)


class RandomUNKWithInputMask(SimpleRandomUNK):

    def __call__(self,
                 input: str,
                 input_mask: Optional[Union[List[int],
                                            List[bool], np.ndarray]] = None,
                 compensate: bool = True) -> str:
        """Place `<UNK>` token at randomly selected words with the probability `unk_ratio`
        only if the corresponding `input_mask` position is marked with `False` or `0`.

        If `compensate == True`, then `unk_ratio` will increase by the expected `compentation_rate`
        calculated by `sum(input_mask)/len(input_mask)`.

        Args:
        * input: str (input string)
        * input_mask: list of integers or booleans, np.ndarray = None
        * compensate: bool = False

        Returns:
        * processed_string: str
        """

        word_list = input.split()

        if input_mask is not None:
            if len(input_mask) < len(word_list):
                input_mask += [1] * (len(word_list) - len(input_mask))

            if compensate:
                compensation_rate = sum(input_mask)/len(input_mask)
                unk_ratio = (1 + compensation_rate) * self.unk_ratio
            else:
                unk_ratio = self.unk_ratio

        else:
            unk_ratio = self.unk_ratio

        mask = np.random.rand(len(word_list)) < unk_ratio

        if input_mask is not None:
            new_list = [self.unk_token if m and not i else word for word, m, i
                        in zip(word_list, mask, input_mask)]
        else:
            new_list = [self.unk_token if m else word for word, m
                        in zip(word_list, mask)]

        return " ".join(new_list)


class UNKWithInputMask(Augmentation):

    def __init__(self,
                 unk_token: str):
        self.unk_token = unk_token

    def __call__(self,
                 input: str,
                 input_mask: Optional[Union[List[int], List[bool], np.ndarray]] = None) -> str:
        """Place `<UNK>` token at all input_mask positions. 
        Apply mask if a corresponding element is `False` or `0`.

        Args:
        * input: str (input string)
        * input_mask: list of integers or booleans, np.ndarray

        Returns:
        * processed_string: str
        """
        word_list = input.split()
        if input_mask is not None:
            new_list = [self.unk_token if not m else word for word, m
                        in zip(word_list, input_mask)]
        else:
            new_list = word_list

        return " ".join(new_list)
