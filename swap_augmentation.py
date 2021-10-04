import os
import pandas as pd


SEED = 42
df = pd.read_csv("/opt/ml/dataset/train/train.csv")

def swap_entity(dataset:pd.DataFrame, col1="subject_entity", col2="object_entity") -> pd.DataFrame:
    '''
    dataset의 "subject_entity"와 "object_entity"를 변경(swap)한 new_dataset을 return
    '''
    new_dataset = dataset.copy()
    new_dataset.rename(columns={col1:col2,col2:col1}, inplace=True) 
    return new_dataset
    
def select_df_by_label(dataset:pd.DataFrame, label: str) -> pd.DataFrame:
    '''
    dataset에서 label에 해당하는 데이터들만 모아 새로운 dataset을 만들어 return
    '''
    new_dataset = dataset.copy()
    new_dataset = new_dataset.loc[dataset['label'] == label, :]
    return new_dataset

def give_data(from_:pd.DataFrame, to_):
    '''
    from_ pd.DataFrame에서 to_ pd.DataFrame으로 넘겨줄 데이터를 길이 비교를 통해 선정 후 return
    '''
    num = (len(from_) - len(to_)) // 2
    select = from_.sample(n=num, replace= False, random_state = SEED)
    return select

def change_label(dataset, label: str):
    '''
    dataset의 원래 라벨을 input으로 들어온 label로 바꾼 후 바뀐 dataset을 return 
    '''
    dataset["label"] = [label] * len(dataset)
    return dataset

# entity 변경 후 증강
aug_labels = ["org:alternate_names", "per:colleagues", "per:alternate_names", "per:spouse", "per:siblings", "per:other_family"]
temp_df = df.copy()
temp_df = temp_df.loc[df['label'].isin(aug_labels), :]
temp_df = swap_entity(temp_df)
new_df = pd.concat([df, temp_df], axis=0)

# 큰 df에서 작은 df로 label 변경후 전달
df_org_member_of = select_df_by_label(df, 'org:member_of')
df_org_members = select_df_by_label(df, 'org:members')

select = give_data(df_org_member_of, df_org_members)
dropped = new_df.drop(select['id'])
add_df = swap_entity(select)
add_df = change_label(add_df,'org:members')
new_df = pd.concat([dropped, add_df], axis=0)

# entity 변경 -> 라벨 변경 -> 원래 데이터에 만든 데이터 추가(concat)
df_per_parents = select_df_by_label(df,'per:parents')
df_per_children = select_df_by_label(df,'per:children')

df_per_parents = swap_entity(df_per_parents)
df_per_parents = change_label(df_per_parents, "per:children")
new_df = pd.concat([new_df, df_per_parents], axis=0)

df_per_children = swap_entity(df_per_children)
df_per_children = change_label(df_per_children, "per:parents")
new_df = pd.concat([new_df, df_per_children], axis=0)
new_df.reset_index(inplace=True, drop=True)

# 새로운 csv로 저장
os.makedirs(f"/opt/ml/dataset/swap_dataset", exist_ok=True)
os.makedirs(f"/opt/ml/dataset/swap_dataset/train", exist_ok=True)
new_df.to_csv("/opt/ml/dataset/swap_dataset/train/train.csv", header=True, index=False)
