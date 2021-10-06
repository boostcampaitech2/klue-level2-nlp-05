import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split


np.random.seed(42)
PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3
HASH_VALUE_SUB = '@#$%@@#$%#'
HASH_VALUE_OBJ = '&$^&#$%^#@'

# entity HSASH_VALUE로 바꾸기, [start_idx:end_idx+1]까지의 글자를 바꿈
def encode_words(sentence:str, start_s, end_s, start_o, end_o) -> str:
    if start_s > start_o:
        sentence = sentence[:start_s] + HASH_VALUE_SUB + sentence[end_s+1:]
        sentence = sentence[:start_o] + HASH_VALUE_OBJ + sentence[end_o+1:]
    else :
        sentence = sentence[:start_o] + HASH_VALUE_OBJ + sentence[end_o+1:]
        sentence = sentence[:start_s] + HASH_VALUE_SUB + sentence[end_s+1:]    
    return sentence


def change_index(sentence, word_s, word_o, len_s, len_o, type_s, type_o):
    '''
    증강된 문장의 개체 index를 재설정
    '''
    sub_index = []
    obj_index = []
    sub_index.append(str(sentence.find(HASH_VALUE_SUB)))
    sub_index.append(str(sentence.find(HASH_VALUE_SUB)+len_s-1))
    sentence = sentence.replace(HASH_VALUE_SUB,word_s, 1)
    
    obj_index.append(str(sentence.find(HASH_VALUE_OBJ)))
    obj_index.append(str(sentence.find(HASH_VALUE_OBJ)+len_o-1))
    sentence = sentence.replace(HASH_VALUE_OBJ,word_o, 1)
    
    entity_sub = "{'word': '"+word_s+"', 'start_idx': "+sub_index[0]+", 'end_idx': "+sub_index[1]+", 'type': '"+type_s+"'}"
    entity_obj = "{'word': '"+word_o+"', 'start_idx': "+obj_index[0]+", 'end_idx': "+obj_index[1]+", 'type': '"+type_o+"'}"

    return sentence, entity_sub, entity_obj



def insert_punctuation(sentence:str, punc_ratio=PUNC_RATIO):
    '''
    ratio만큼 PUNCTATIONS를 sentence에 랜덤으로 추가
    '''
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1)) 
    qs = random.sample(range(0, len(words)), q)
    
    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)

    return new_line


def insert_punc_and_change_index(data_row, punc_ratio=PUNC_RATIO):
    '''
    encode_words, insert_punctuation, change_index, merge to pd.Series
    '''
    # parse word, idx, len, type
    word_s, start_s, end_s, type_s = list(eval(data_row["subject_entity"]).values())
    word_o, start_o, end_o, type_o = list(eval(data_row["object_entity"]).values())

    len_s = len(word_s)
    len_o = len(word_o)
    
    # input 원래 sentence -> output entity가 encode된 sentence
    encoded_sentence = encode_words(data_row['sentence'], start_s, end_s, start_o, end_o) 
    
    new_sentence = insert_punctuation(encoded_sentence) 
    
    new_sentence, entity_sub, entity_obj = change_index(new_sentence, word_s, word_o, len_s, len_o, type_s, type_o)  
    
    data_row['sentence'] = new_sentence
    data_row['subject_entity'] = entity_sub
    data_row['object_entity'] = entity_obj

    return data_row


def main(data_train:pd.DataFrame, data_val:pd.DataFrame, aug):
    '''
    AEDA 데이터 증강을 위한 순서를 담은 함수
    '''
    aug_train = pd.DataFrame()
    aug_val = pd.DataFrame()

    for _ in range(aug):  # 들어온 label별 반복 횟수
        train_new = data_train.apply(lambda x: insert_punc_and_change_index(x), axis=1) 
        val_new = data_val.apply(lambda x: insert_punc_and_change_index(x), axis=1)  

        aug_train = pd.concat([aug_train, train_new], axis=0)
        aug_val = pd.concat([aug_val, val_new], axis=0)

        aug_train.reset_index(inplace=True, drop=True)
        aug_val.reset_index(inplace=True, drop=True)

    return aug_train, aug_val


def iterate_main(path:str, min_num=300):
    '''
    split train, val set and duplicate if needed
    '''
    df = pd.read_csv(path)
    label_num_list = [(i, min_num // j) for i, j in zip(df["label"].value_counts().index, df["label"].value_counts())]

    final_df_train = pd.DataFrame()
    final_df_val = pd.DataFrame()
    for label, iter_num in label_num_list:
        df_label = df.loc[df['label'] == label, :]
        df_train, df_val = train_test_split(df_label, test_size=0.2, random_state=42)
        if iter_num != 0:
            df_train, df_val = main(df_train, df_val, iter_num)
        
        final_df_train = pd.concat([final_df_train, df_train], axis=0)
        final_df_val = pd.concat([final_df_val, df_val], axis=0)

    os.makedirs(f"/opt/ml/dataset/aeda_bal{min_num}_dataset/", exist_ok=True)
    os.makedirs(f"/opt/ml/dataset/aeda_bal{min_num}_dataset/train", exist_ok=True)
    
    final_df_train.to_csv(f"/opt/ml/dataset/aeda_bal{min_num}_dataset/train/train.csv", header=True, index=False)
    final_df_val.to_csv(f"/opt/ml/dataset/aeda_bal{min_num}_dataset/train/valid.csv", header=True, index=False)

iterate_main("/opt/ml/dataset/train/train.csv", 300)
iterate_main("/opt/ml/dataset/train/train.csv", 500)
