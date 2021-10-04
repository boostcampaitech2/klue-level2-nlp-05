import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

random.seed(42)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
NUM_AUGS = [1, 2, 4, 8]
PUNC_RATIO = 0.3


def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
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
	
def main(dataset:str):
	'''
	dataset을 특정 횟수만큼 증강 후 추가
	dataset에 insert_punctuation_marks를 이용하여 만든 sentence(기본 sentence만 바꾸고 나머지 정보는 그대로)를 추가
	'''
	orig_df = pd.read_csv(dataset)
	for aug in tqdm(NUM_AUGS):
		result_aug = orig_df.copy()
		for _ in range(aug):
			df_aug = orig_df.copy()
			df_aug['sentence'] = orig_df["sentence"].map(insert_punctuation_marks)
			result_aug = pd.concat([result_aug, df_aug], axis=0)
			result_aug.reset_index(inplace=True,drop=True)

		os.makedirs(f"/opt/ml/dataset/aeda_{aug}_dataset/", exist_ok=True)
		os.makedirs(f"/opt/ml/dataset/aeda_{aug}_dataset/train", exist_ok=True)
		result_aug.to_csv(f"/opt/ml/dataset/aeda_{aug}_dataset/train/train.csv", header=True, index=False)


if __name__ == "__main__":
	main("/opt/ml/dataset/train/train.csv")
