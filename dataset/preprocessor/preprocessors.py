from abc import abstractmethod
import pandas as pd
import pickle as pickle


def num_to_label(num_label):
    origin_label = []
    with open('/opt/ml/code/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in num_label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def label_to_num(origin_label):
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in origin_label:
        num_label.append(dict_label_to_num[v])

    return num_label


class Preprocessor:

    @abstractmethod
    def __call__(self, dataset) -> pd.DataFrame:
        return dataset


class BaselinePreprocessor(Preprocessor):

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        subject_entity = []
        object_entity = []
        concat_entity = []

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


class ExtendedPreprocessor(Preprocessor):

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """subject_entity와 object_entity를 확장한 DataFrame으로 변경"""
        new_data = {
            'id': [],
            'sentence': [],
            'sentence_length': [],
            'subject_entity_word': [],
            'subject_entity_start_idx': [],
            'subject_entity_end_idx': [],
            'subject_entity_type': [],
            'object_entity_word': [],
            'object_entity_start_idx': [],
            'object_entity_end_idx': [],
            'object_entity_type': [],
            'label': [],
            'source': []
        }

        for i, row in data.iterrows():
            subject_dict = eval(row['subject_entity'])
            object_dict = eval(row['object_entity'])

            new_data['id'].append(row['id'])
            new_data['sentence'].append(row['sentence'])
            new_data['sentence_length'].append(len(row['sentence']))
            new_data['subject_entity_word'].append(subject_dict['word'])
            new_data['subject_entity_start_idx'].append(subject_dict['start_idx'])
            new_data['subject_entity_end_idx'].append(subject_dict['end_idx'])
            new_data['subject_entity_type'].append(subject_dict['type'])
            new_data['object_entity_word'].append(object_dict['word'])
            new_data['object_entity_start_idx'].append(object_dict['start_idx'])
            new_data['object_entity_end_idx'].append(object_dict['end_idx'])
            new_data['object_entity_type'].append(object_dict['type'])
            new_data['label'].append(row['label'])
            new_data['source'].append(row['source'])

        return pd.DataFrame(new_data)


class T5BasicPreprocessor(ExtendedPreprocessor):

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:

        new_df = super(T5BasicPreprocessor, self).__call__(data)

        t5_inputs = []
        task_description = "klue_re text: "

        for i, row in new_df.iterrows():
            sentence = row['sentence']

            sbj_from = row['subject_entity_start_idx']
            sbj_to = row['subject_entity_end_idx'] + 1
            obj_from = row['object_entity_start_idx']
            obj_to = row['object_entity_end_idx'] + 1

            if sbj_from < obj_from:
                new_sentence = task_description + sentence[:sbj_from] \
                    + "*" + sentence[sbj_from:sbj_to] + "*" \
                    + sentence[sbj_to:obj_from] \
                    + "#" + sentence[obj_from:obj_to] + "#" \
                    + sentence[obj_to:]
            else:
                new_sentence = task_description + sentence[:obj_from] \
                    + "*" + sentence[obj_from:obj_to] + "*" \
                    + sentence[obj_to:sbj_from] \
                    + "#" + sentence[sbj_from:sbj_to] + "#" \
                    + sentence[sbj_to:]


            # input format: "klue re: ~~~*{subject}*~~~#{object}#~~~"
            # refers to: https://github.com/AIRC-KETI/ke-t5

            t5_inputs.append(new_sentence)
            if i < 5:
                print("Old:", sentence)
                print("New:", new_sentence)

        new_df['t5_inputs'] = t5_inputs
        new_df['label'] = label_to_num(new_df['label'].tolist())

        return new_df
