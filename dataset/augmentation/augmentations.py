from abc import abstractmethod
from typing import Optional, Union, List

import numpy as np

import re
import random

import fasttext
import fasttext.util
from konlpy.tag import Okt

class Augmentation:

    @abstractmethod
    def __init__(self, tokenizer):
        if tokenizer is not None:
            self.unk_token = tokenizer.unk_token
        pass

    @abstractmethod
    def __call__(self, input_text: str) -> str:
        pass


class SimpleRandomUNK(Augmentation):

    def __init__(self, tokenizer, unk_ratio: float = 0.1):

        self.unk_ratio = unk_ratio
        self.unk_token = tokenizer.unk_token

    def __call__(self, input_text: str) -> str:
        """Place `<UNK>` token at randomly selected words with the probability `unk_ratio`.

        Args:
        * input: str (input string)

        Returns:
        * processed_string: str
        """
        word_list = input_text.split()
        mask = np.random.rand(len(word_list)) < self.unk_ratio
        new_list = [self.unk_token if m else word for word, m
                    in zip(word_list, mask)]

        return " ".join(new_list)


class RandomUNKWithInputMask(SimpleRandomUNK):

    def __call__(self,
                 input_text: str,
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

        word_list = input_text.split()

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

    def __init__(self, tokenizer):
        self.unk_token = tokenizer.unk_token

    def __call__(self,
                 input_text: str,
                 input_mask: Optional[Union[List[int], List[bool], np.ndarray]] = None) -> str:
        """Place `<UNK>` token at all input_mask positions. 
        Apply mask if a corresponding element is `False` or `0`.

        Args:
        * input: str (input string)
        * input_mask: list of integers or booleans, np.ndarray

        Returns:
        * processed_string: str
        """
        word_list = input_text.split()

        if input_mask is not None:
            if len(input_mask) < len(word_list):
                # dealing with inherent noise in text data
                input_mask += [1] * (len(word_list) - len(input_mask))

            new_list = [self.unk_token if not m else word for word, m
                        in zip(word_list, input_mask)]
        else:
            new_list = word_list

        return " ".join(new_list)


class RandomReplaceWords(Augmentation):

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)

        self.ft = fasttext.load_model('fasttext/cc.ko.100.bin')
        self.okt = Okt()

    def get_nearest_word_vectors(self, query_word: str, k: int = 10):
        results = self.ft.get_nearest_neighbors(query_word, k)
        word = [query_word]
        similarity = [1]
        word_vectors = [self.ft.get_word_vector(query_word)]
        for d, w in results:
            word.append(w)
            similarity.append(d)
            word_vectors.append(self.ft.get_word_vector(w))
        return {'word': word,
                'similarity': similarity,
                'word_vectors': word_vectors}


    def change_random_word(self, sentence: str, min_changes: int = 0, max_changes: int = 3):
        
        nouns  = self.okt.nouns(sentence)
        
        num_changes = min(random.randint(min_changes, max_changes), len(nouns))
        cnt = 0
        trials = 0

        while cnt < num_changes and trials < 10:

            length = 0
            trials += 1

            while length < 2:
                choice = random.randint(0, len(nouns)-1)
                original_word = nouns[choice]
                length = len(original_word)

            start_idx = sentence.find(original_word)
            end_idx = start_idx + len(original_word)

            if start_idx == -1:
                continue

            k = 5
            results = self.get_nearest_word_vectors(original_word, k=k)

            for i in range(1, k+1):
                replacing_word = results['word'][i]
                tokenized = self.okt.nouns(replacing_word)
                if len(tokenized) == 0:
                    break
                replacing_word = tokenized[0]

                if len(original_word) < 2:
                    break
                if original_word == replacing_word:
                    continue
                if abs(len(replacing_word) - len(original_word)) > 2:
                    continue
                if re.match(r'[^ㄱ-ㅣ가-힣]+', replacing_word):
                    continue

                sentence = sentence[:start_idx] + replacing_word + sentence[end_idx:]
                cnt += 1
                # print("changed from", original_word, "to", replacing_word)
                break

        return sentence

    def __call__(self, input_text: str):
        RATIO = 0.2
        if random.random() < RATIO:
            return self.change_random_word(input_text)
        else:
            return input_text


class RandomFlip(Augmentation):

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)

    def __call__(self, input_text: str) -> str:

        PROB = 0.2
        word_list = input_text.split()

        if random.random() < PROB:
        
            flip_point = random.randint(int(len(word_list)*0.2), int(len(word_list)*0.8))

            prev_last_word = word_list[-1]
            prev_last_word = prev_last_word[:-1] + ","
            # remove last punctuation and add ","
            word_list[-1] = prev_last_word

            new_last_word = word_list[flip_point]
            new_last_word = new_last_word + "."
            word_list[flip_point] = new_last_word

            word_list = word_list[flip_point+1:] + word_list[0:flip_point+1]

        return " ".join(word_list)

    