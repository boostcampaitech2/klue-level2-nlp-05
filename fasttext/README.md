# FASTTEXT

## Introduction

Fasttext는 Facebook에서 공개한 efficient learning of word representation이자 pre-trained word vector representation입니다. 영어뿐만 아니라 157개 언어의 pre-trained representation을 제공합니다.

## How to use

일단 fasttext로 사전훈련된 모델을 로컬에 다운로드 해야 합니다. 이는 아래 명령어를 실행하면 됩니다.

```bash
python download_fasttext.py
```

기본적인 모델은 300차원으로 이루어져 있습니다. 따라서 유사 단어 검색 등에 있어서 kNN 알고리즘을 사용하므로 연산량이 상당합니다. 따라서 다운로드된 모델을 100차원으로 축소하여 진행합니다.

```bash
python reduce_dim.py cc.ko.300.bin cc.ko.100.bin 100
```

이후 자세한 내용은 fasttext.ipynb를 참고하면 됩니다.

## Data Augmentation

Fasttext를 이용하여 data augmentation을 수행합니다.

### 🚧🚧🚧 Under Construction 🚧🚧🚧