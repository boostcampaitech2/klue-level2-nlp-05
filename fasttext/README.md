# fastText

"Library for efficient text classification and representation learning"

P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

E. Grave*, P. Bojanowski*, P. Gupta, A. Joulin, T. Mikolov, Learning Word Vectors for 157 Languages

## Introduction

fastText는 Facebook에서 공개한 efficient learning of word representation이자 pre-trained word vector representation입니다. 영어뿐만 아니라 157개 언어의 pre-trained representation을 제공합니다.

fastText는 skip-gram 방식과 CBOW 방식을 모두 지원합니다. 

![cbow&skipgram](https://fasttext.cc/img/cbo_vs_skipgram.png)

## Installation

일단, 기본적인 세팅이 필요합니다. 현 도커 환경에서 c++11 이상을 지원하는 gcc가 설치되어 있지 않습니다. 따라서 일단 아래 명령어를 실행해서 컴파일러를 설치해줍니다.

```bash
sudo apt install make
sudo apt-get install build-essential -y
sudo apt install cmake
```

또한, 기본적인 환경에서는 matplotlib가 한국어 폰트를 지원하지 않습니다. 따라서 네이버 나눔폰트를 설치해야 합니다.

```bash
apt-get install fontconfig
apt-get install fonts-nanum*
```

이후 matplotlib의 캐시를 초기화해서, 시스템에 설치된 폰트를 다시 받아가도록 합니다.

```bash
rm -rf ~/.cache/matplotlib/*
```

나눔폰트를 사용하고자 한다면, 주피터 노트북에서 아래와 같이 설정해주면 됩니다.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'
```

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

Fasttext를 이용하여 data augmentation을 수행합니다. 기본적으로는 `ft.get_word_vector("word": str)`과 `ft.get_nearest_neighbors("word": str, k: int)` 함수를 이용합니다. 또한, `ft.get_analogies("a", "b", "c")`를 사용하는 것도 고려해볼 수 있습니다. `ft.get_analogies()` 함수는 word2vec에서 흔히 사용되는 A - B + C 형태의 추론을 사용합니다.

### 🚧🚧🚧 Under Construction 🚧🚧🚧

## References

https://fasttext.cc

https://inahjeon.github.io/fasttext/