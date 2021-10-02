from abc import abstractmethod
import pandas as pd
import pickle as pickle


def label_to_num(label):
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

class Preprocessor:

    @abstractmethod
    def __call__(self, dataset) -> pd.DataFrame:
        return dataset


class BaselinePreprocessor(Preprocessor):

    def __call__(self, data: pd.DataFrame):
        subject_entity = []
        object_entity  = []
        concat_entity  = []

        # TODO: work on parsing
        for i, j in zip(data['subject_entity'], data['object_entity']):
            sbj_word = i[1:-1].split(',')[0].split(':')[1]
            obj_word = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(sbj_word)
            object_entity.append(obj_word)
            concat_entity.append(sbj_word + '[SEP]' + obj_word)

        new_df = pd.DataFrame({'id': data['id'],
                               'sentence': data['sentence'],
                               'subject_entity': subject_entity,
                               'object_entity': object_entity,
                               'concat_entity': concat_entity,
                               'label': label_to_num(data['label']),
                               'source': data['source']})

        return new_df

class T5BasicPreprocessor(Preprocessor):

    def __call__(self, data: pd.DataFrame):
        subject_entity = []
        object_entity  = []
        concat_entity  = []

        # TODO: work on parsing
        for i, j in zip(data['subject_entity'], data['object_entity']):
            sbj_word = i[1:-1].split(',')[0].split(':')[1]
            obj_word = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(sbj_word)
            object_entity.append(obj_word)
            concat_entity.append("Guess relation from " + sbj_word + " to " + obj_word +":")

        new_df = pd.DataFrame({'id': data['id'],
                               'sentence': data['sentence'],
                               'subject_entity': subject_entity,
                               'object_entity': object_entity,
                               'concat_entity': concat_entity,
                               'label': label_to_num(data['label']),
                               'source': data['source']})

        return new_df