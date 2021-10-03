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
    
    def __call__(self,dataset):
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
                'concat_entity': [],
                'label': [],
                'source': []
            }

        for i, row in dataset.iterrows():
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
            new_data['concat_entity'].append(subject_dict['word']+'[SEP]'+object_dict['word'])
            new_data['label'].append(row['label'])
            new_data['source'].append(row['source'])

        new_data['label'] = label_to_num(new_data['label'])
        return pd.DataFrame(new_data)

        
class EntitySpecialTokenPreprocessor(ExtendedPreprocessor):

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        
        dataset = super(EntitySpecialTokenPreprocessor, self).__call__(data)
        
        entity_special_tokens = []
        entity_sentences = []
        subject_entity_start_tokens = []
        subject_entity_end_tokens = []
        object_entity_start_tokens = []
        object_entity_end_tokens = []

        for i, row in dataset.iterrows():
            s_entity = {
                'prefix': 'S',
                'word': row['subject_entity_word'],
                's_idx': row['subject_entity_start_idx'],
                'e_idx': row['subject_entity_end_idx'],
                'type': row['subject_entity_type']
            }
            o_entity = {
                'prefix': 'O',
                'word': row['object_entity_word'],
                's_idx': row['object_entity_start_idx'],
                'e_idx': row['object_entity_end_idx'],
                'type': row['object_entity_type']
            }

            entities = sorted([s_entity, o_entity], key=lambda item: item['s_idx'], reverse=True)

            special_tokens = []
            for entity in entities:
                s_token = f"[{entity['prefix']}:{entity['type']}]"  # ex) "[S:ORG]"
                e_token = f"[/{entity['prefix']}:{entity['type']}]"  # ex) "[/S:ORG]"
                
                # entity: dict_비틀즈
                entity['s_token'] = s_token
                entity['e_token'] = e_token
                
                if entity['prefix'] == 'S':
                    subject_entity_start_tokens.append(s_token)
                    subject_entity_end_tokens.append(e_token)
                else:
                    object_entity_start_tokens.append(s_token)
                    object_entity_end_tokens.append(e_token)
                
                special_tokens.extend([s_token, e_token])
            # special_tokens = ["[S : ORG ]", [S : ORG ], [O : PER ], [O : PER ]]

            # 전체 스페셜 토큰 리스트에 추가
            for special_token in special_tokens:
                if special_token not in entity_special_tokens:
                    entity_special_tokens.append(special_token)

            sentence = row['sentence']
            for entity in entities:
                s_idx, e_idx = entity['s_idx'], entity['e_idx']
                new_sentence = ''
                new_sentence += sentence[:s_idx]
                new_sentence += entity['s_token']
                new_sentence += entity['word']
                new_sentence += entity['e_token']
                new_sentence += sentence[e_idx+1:]
                sentence = new_sentence

            entity_sentences.append(sentence)

        dataset['sentence'] = entity_sentences
        dataset['subject_entity_start_token'] = subject_entity_start_tokens
        dataset['subject_entity_end_token'] = subject_entity_end_tokens
        dataset['object_entity_start_token'] = object_entity_start_tokens
        dataset['object_entity_end_token'] = object_entity_end_tokens  

        concat_entity = []

        for i, row in dataset.iterrows():
            temp = row['subject_entity_start_token'] + \
                row['subject_entity_word'] + \
                row['subject_entity_end_token'] + \
                '[SEP]' + \
                row['object_entity_start_token'] + \
                row['object_entity_word'] + \
                row['object_entity_end_token']
            concat_entity.append(temp) 

        dataset["concat_entity"] = concat_entity

        return pd.DataFrame(dataset), entity_special_tokens


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
