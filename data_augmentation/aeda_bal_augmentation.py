import os
import numpy as np
import pandas as pd


np.random.seed(42)
PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3
HASH_VALUE_SUB = '@#$%@@#$%#'
HASH_VALUE_OBJ = '&$^&#$%^#@'
TRAIN_PATH = "/opt/ml/dataset/train/train.csv"
DATASET_PATH = "/opt/ml/dataset"


def encode_words(sentence, start_s, end_s, start_o, end_o):
    '''
    entity를 HASH_VALUE로 대체, [start_idx:end_idx+1]까지의 글자를 바꿈
    '''
    if start_s > start_o:
        sentence = sentence[:start_s] + HASH_VALUE_SUB + sentence[end_s+1:]
        sentence = sentence[:start_o] + HASH_VALUE_OBJ + sentence[end_o+1:]
    else :
        sentence = sentence[:start_o] + HASH_VALUE_OBJ + sentence[end_o+1:]
        sentence = sentence[:start_s] + HASH_VALUE_SUB + sentence[end_s+1:]    
    
    return sentence


def insert_punctuation(sentence, punc_ratio=PUNC_RATIO) :
    '''
    ratio만큼 PUNCTATIONS를 sentence에 랜덤으로 추가
    '''
    words = sentence.split(' ')
    new_line = []
    q = np.random.randint(1, int(punc_ratio * len(words) + 1)) 
    qs = np.random.choice(range(0, len(words)), q)
        
    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[np.random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)

    return new_line


def change_index(sentence, word_s, word_o, len_s, len_o, type_s, type_o):
    '''
    증강된 문장의 개체 index를 재설정
    '''
    sub_index = []
    obj_index = []
    
    sub_index.append(str(sentence.find(HASH_VALUE_SUB)))
    sub_index.append(str(sentence.find(HASH_VALUE_SUB) + len_s - 1))
    sentence = sentence.replace(HASH_VALUE_SUB, word_s, 1)
    
    obj_index.append(str(sentence.find(HASH_VALUE_OBJ)))
    obj_index.append(str(sentence.find(HASH_VALUE_OBJ) + len_o - 1))
    sentence = sentence.replace(HASH_VALUE_OBJ, word_o, 1)
    
    entity_sub = "{'word': '" + word_s + "', 'start_idx': " + sub_index[0] + ", 'end_idx': " + sub_index[1] + ", 'type': '" + type_s + "'}"
    entity_obj = "{'word': '" + word_o + "', 'start_idx': " + obj_index[0] + ", 'end_idx': " + obj_index[1] + ", 'type': '" + type_o + "'}"

    return sentence, entity_sub, entity_obj
    

def insert_punc_and_change_index(data_row):
    '''
    encode_words, insert_punctuation, change_index, merge to pd.Series
    '''
    word_s, start_s, end_s, type_s = list(eval(data_row["subject_entity"]).values())
    word_o, start_o, end_o, type_o = list(eval(data_row["object_entity"]).values())
    len_s = len(word_s)
    len_o = len(word_o)

    encoded_sentence = encode_words(data_row['sentence'], start_s, end_s, start_o, end_o) 
    new_sentence = insert_punctuation(encoded_sentence) 
    new_sentence, entity_sub, entity_obj = change_index(new_sentence, word_s, word_o, len_s, len_o, type_s, type_o)  
    
    data_row['sentence'] = new_sentence
    data_row['subject_entity'] = entity_sub
    data_row['object_entity'] = entity_obj

    return data_row
    

def main(dataset, aug):
    '''
    주어진 dataframe에 AEDA 데이터 증강을 적용하기 위한 함수
    '''
    result_aug = pd.DataFrame()
    for _ in range(aug):
        df_aug = dataset.copy()
        new = df_aug.apply(lambda x: insert_punc_and_change_index(x), axis=1)
        result_aug = pd.concat([result_aug, new], axis=0)
        result_aug.reset_index(inplace=True, drop=True)

    return result_aug


def iterate_main(path, min_num=300):
    '''
    data imbalance를 해소하기 위해 특정 라벨 데이터를 최소 수준(min_num)까지 증강
    '''
    df = pd.read_csv(path)
    label_num_list = [(i, min_num // j) for i, j in zip(df["label"].value_counts().index, df["label"].value_counts())]
    for label,iter_num in label_num_list:
        if iter_num == 0:
            continue
        new_df = df.loc[df['label'] == label, :]
        result_df = main(new_df, iter_num)
        df = pd.concat([df,result_df], axis=0)

    os.makedirs(f"{DATASET_PATH}/aeda_bal{min_num}_dataset/", exist_ok=True)
    os.makedirs(f"{DATASET_PATH}/aeda_bal{min_num}_dataset/train", exist_ok=True)
    df.to_csv(f"{DATASET_PATH}/aeda_bal{min_num}_dataset/train/train.csv", header=True, index=False)


if __name__ == "__main__":
    iterate_main(TRAIN_PATH, 300)
    iterate_main(TRAIN_PATH, 500)
    iterate_main(TRAIN_PATH, 1000)
