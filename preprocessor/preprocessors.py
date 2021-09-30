from abc import abstractmethod
from typing_extensions import Concatenate
import pandas as pd


class Preprocessor:

    @abstractmethod
    def __call__(self, dataset) -> pd.DataFrame:
        return dataset


class BaselinePreprocessor(Preprocessor):

    def __call__(self, dataset: pd.DataFrame):
        subject_entity = []
        object_entity  = []
        concat_entity  = []

        sbj_start_idx = []
        obj_start_idx = []

        sbj_end_idx = []
        obj_end_idx = []
        
        subject_type = []
        object_type  = []

        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            sbj_word = i[1:-1].split(',')[0].split(':')[1]
            obj_word = j[1:-1].split(',')[0].split(':')[1]

            sbj_start_idx = int(i[1:-1].split(',')[1].split(':')[1])
            obj_start_idx = int(j[1:-1].split(',')[1].split(':')[1])

            sbj_end_idx  = int(i[1:-1].split(',')[2].split(':')[1])
            obj_end_idx  = int(j[1:-1].split(',')[2].split(':')[1])

            sbj_type = i[1:-1].split(',')[3].split(':')[1]
            obj_type = i[1:-1].split(',')[3].split(':')[1]

            subject_entity.append(sbj_word)
            object_entity.append(obj_word)
            concat_entity.append(sbj_word + '[SEP]' + obj_word)

        dataset = pd.DataFrame({'id': dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity': subject_entity,
                                'sbj_start_idx': sbj_start_idx,
                                'sbj_end_idx': sbj_end_idx,
                                'subject_type': sbj_type,
                                'object_entity': object_entity,
                                'obj_start_idx': obj_start_idx,
                                'obj_end_idx': obj_end_idx,
                                'object_type': obj_type,
                                'concat_entity': concat_entity,
                                'label': dataset['label'],
                                'source': dataset['source']})

        return dataset
