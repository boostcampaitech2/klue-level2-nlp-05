import pandas as pd
from tqdm import tqdm
import pickle

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in tqdm(zip(dataset['subject_entity'], dataset['object_entity']), total=len(dataset)):
    i = eval(i)['word']
    j = eval(j)['word']

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def new_preprocessing_dataset(dataset):
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

    for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
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

def load_train_data():
    """학습 데이터셋을 불러온다."""
    pd_dataset = pd.read_csv("/opt/ml/dataset/train/train.csv")
    dataset = preprocessing_dataset(pd_dataset)
    return dataset

def load_new_train_data():
    """확장된 학습 데이터셋을 불러온다."""
    pd_dataset = pd.read_csv("/opt/ml/dataset/train/train.csv")
    dataset = new_preprocessing_dataset(pd_dataset)
    return dataset

def label_to_num(label):
    """label 텍스트를 인덱스로 변환"""
    num_label = []
    with open('/opt/ml/boostcamp-klue/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label

def num_to_label(label):
    """label 인덱스를 텍스트로 변환"""
    origin_label = []
    with open('/opt/ml/boostcamp-klue/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    return origin_label    
