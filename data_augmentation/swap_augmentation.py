import os
import pandas as pd
import numpy as np

np.random.seed(42)
TRAIN_PATH = "/opt/ml/dataset/train/train.csv"
DATASET_PATH = "/opt/ml/dataset"

df = pd.read_csv(TRAIN_PATH)


def swap_entity(dataset, col1="subject_entity", col2="object_entity"):
    '''
    dataset의 "subject_entity"와 "object_entity"를 변경(swap)한 new_dataset을 return
    '''
    new_dataset = dataset.copy()
    new_dataset.rename(columns={col1:col2,col2:col1}, inplace=True) 
    return new_dataset
    
    
def select_df_by_label(dataset, label):
    '''
    dataset에서 label에 해당하는 데이터들만 모아 새로운 dataset을 만들어 return
    '''
    new_dataset = dataset.copy()
    new_dataset = new_dataset.loc[dataset['label'] == label, :]
    return new_dataset


def give_data(large_df, small_df):
    '''
    large_df에서 small_df으로 넘겨줄 데이터를 선정 후 return
    '''
    num = (len(large_df) - len(small_df)) // 2
    select = large_df.sample(n=num, replace=False)
    return select


def change_label(dataset, label):
    '''
    dataset의 원래 라벨을 input으로 들어온 label로 바꾼 후 바뀐 dataset을 return 
    '''
    dataset["label"] = [label] * len(dataset)
    return dataset


# "Entity-swappable": entity만 변경
aug_labels = ["org:alternate_names", "per:colleagues", "per:alternate_names", "per:spouse", "per:siblings", "per:other_family"]
temp_df = df.copy()
temp_df = temp_df.loc[df['label'].isin(aug_labels), :]
temp_df = swap_entity(temp_df)
new_df = pd.concat([df, temp_df], axis=0)

# "Label-swappable": entity 변경 후 개수가 많은 label의 데이터셋 일부를 적은 label의 데이터셋으로 변경
df_org_member_of = select_df_by_label(df, 'org:member_of')
df_org_members = select_df_by_label(df, 'org:members')

select = give_data(df_org_member_of, df_org_members)
dropped = new_df.drop(select['id'])
add_df = swap_entity(select)
add_df = change_label(add_df,'org:members')
new_df = pd.concat([dropped, add_df], axis=0)

# "Label-swappable": dataframe을 복사 후 entity와 label 모두 변경
df_per_parents = select_df_by_label(df, 'per:parents')
df_per_children = select_df_by_label(df, 'per:children')

df_per_parents = swap_entity(df_per_parents)
df_per_parents = change_label(df_per_parents, "per:children")
new_df = pd.concat([new_df, df_per_parents], axis=0)

df_per_children = swap_entity(df_per_children)
df_per_children = change_label(df_per_children, "per:parents")
new_df = pd.concat([new_df, df_per_children], axis=0)
new_df.reset_index(inplace=True, drop=True)

# 새로운 csv로 저장
os.makedirs(f"{DATASET_PATH}/swap_dataset", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/swap_dataset/train", exist_ok=True)
new_df.to_csv(f"{DATASET_PATH}/swap_dataset/train/train.csv", header=True, index=False)
