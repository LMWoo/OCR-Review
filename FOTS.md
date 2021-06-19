# FOTS: Fast Oriented Text Spotting with a Unified Network

# 논문
- https://arxiv.org/pdf/1704.03155v2.pdf

# 논문 요약
## 1. Introduction

```
자연 이미지에서 텍스트를 읽는 것이 문서 분석, 장면 이해, 로봇 네비게이션, 이미지 검색 등 비전 분야에서 주목을 이끌었다.
이전에 텍스트 감지 및 인식에서 상당한 진전을 이루었지만, 텍스트의 큰 변화와 복잡한 배경때문에 여전히 어렵다.

장면 텍스트 읽기에서 대부분의 방식은 텍스트 감지 및 인식으로 나누는 것이며, 두 개가 개별적으로 다뤄진다.
두 파트에서 딥러닝 기반 방식이 지배적이다.
텍스트 감지에서 보통 합성곱 신경망은 이미지에서 feature map을 추출하는데 쓰이며,
디코더는 텍스트 영역을 디코딩 하는데 쓰인다.
텍스트 인식에서 시계열 예측을 위한 신경망은 텍스트 영역에서 하나씩 수행된다.
이 것은 텍스트 영역이 많은 이미지 일수록 시간이 걸린다.
다른 문제는 텍스트 감지 및 인식에서 공유되는 시각적 단서에서 상관관계를 무시한다.
단일 감지 네트워크는 텍스트 인식 라벨에 대한 supervised learning을 할 수 없으며, 반대도 마찬가지이다.

이 논문에서는 텍스트 감지 및 인식을 동시에 수행하는 것을 제안한다.
이것은 end-to-end방식으로 훈련 할 수 있는 fast oriented text spotting system으로 이어진다.
이전의 2단계 text spotting와 다르게, 이 방식은 합성곱 신경망으로 더 많은 features를 학습하며,
이 features는 텍스트 감지 및 인식 사이에서 공유되며,
이 두 작업 간에 supervised learning을 할 수 있게 된다.
feature추출은 보통 시간이 많이 걸리므로, 단일 텍스트 감지 네트워크로 계산을 줄인다.
텍스트 감지 및 인식을 연결하는 것에 핵심은 RoIRotate이며,
이는 oriented detection bounding boxes에 따라 feature maps의 적합한 feature를 갖는다.
```

<img width="853" alt="스크린샷 2021-06-19 오후 1 35 05" src="https://user-images.githubusercontent.com/80749934/122631044-6ab0a280-d103-11eb-87af-ab6fdbab5971.png">

```
구조는 위 그림과 같다.
feature maps는 먼저 shared convolutions으로 추출된다.
text detection branch는 feature map에 대한 bounding box를 예측한다.
RoIRotate는 feature map에 대한 감지 결과에 해당되는 text proposal features를 추출한다.
그런 다음, 이 text proposal features는 텍스트 인식을 위해 RNN encoder와 CTC decoder로 보내진다.
이 네트워크에서 모든 모듈이 미분가능해서, 전체 시스템은 end-to-end로 학습 될 수 있다.
이 것은 최초의 텍스트 감지 및 인식을 위한 학습 가능한 end-to-end frame work이다.
```

3가지 기여
1. convolutional features를 공유하여, 적은 계산과 실시간으로 텍스트를 감지 및 인식할 수 있다.
2. RoIRotate는 새로운 미분가능한 연산자이며, convolutional feature maps으로 부터 텍스트 영역을 추출한다. 이 연산자는 텍스트 감지 및 인식을 end-to-end pipeline으로 연결한다.
3. ICDAR2015, ICDAR2017 MLT, ICDAR2013을 포함한 텍스트 감지 및 인식 벤치 마크에서 최첨단 방식을 능가한다.

## 2. Related Work

```
text spotting은 비전 및 문서 분석에서 활발한 주제이다.
이 장에서, 텍스트 감지 및 인식 방식과 이 두 개를 결합한 text spotting방식을 포함한 관련 논문을 소개한다.
```

### 2.1 Text Detection

```
대부분 텍스트 감지 방식은 텍스트를 characters의 구성으로 고려한다.
이러한 character 기반 방식은 먼저 characters를 localize하며, 그것을 word또는 text lines로 묶는다.
기존 방식에는 Sliding-window기반 방식 및 connected-components기반 방식이 있다.

최근, 텍스트 감지를 위한 많은 딥러닝 기반 방식이 제안된다.
Tian은 고정된 너비 sequential proposals을 예측 및 연결하는 vertical anchor mechanism을 사용한다.
Ma은 Rotation RPN과 Rotation RoI pooling을 제안하여 새로운 rotation-based framework를 소개한다.
Shi는 text segments를 예측한 다음 linkage prediction을 이용해 완전한 instances로 연결한다.
Zhou는 여러 방향의 텍스트 인식을 위해 dense predictions과 한 단계 후처리로 deep direct regression방식을 소개한다.
```

### 2.2 Text Recognition

```
일반적으로, 장면 텍스트 인식은 다양한 길이의 텍스트 이미지에서 시계열 라벨을 디코딩하는데 초첨을 맞춘다.
대부분 이전 방식은 각 characters를 찾고 잘못 분류 된 것을 제거한다.
character기반 방식이외에, 최근 텍스트 인식 방식은 세가지로 분류 된다.
1. word classification based method
2. sequence-to-label decode based method
3. sequence-to-sequence model based method

Jaderberg는 텍스트 인식을 기존 multi-class분류로써 제안한다.
Su는 텍스트 인식은 sequence labelling으로 고안하며, HOG features에 RNN이 적용되며 CTC를 디코더로 사용한다.
Shi, He는 최대로 CNN features를 인코딩하고 CTC로 디코딩 하는 deep recurrent models을 제안한다.
Lee는 자동적으로 확실한 extracted CNN features에 초점을 맞추고,
암묵적으로 RNN으로 구현된 character level language model을 학습하는 
attention기반 sequence-to-sequence구조를 사용한다.
Shi는 불규칙한 입력이미지를 다루기 위해, 
왜곡된 텍스트 영역을 텍스트 인식에 적합하게 변형하는 spatial attention mechanism을 소개한다.
```

### 2.3 Text Spotting

```
대부분 기존 text spotting방식은 텍스트 감지기로 text proposals을 생성하고, 텍스트 인식기로 그것을 인식한다.
Jaderberg는 ennsemble model을 사용하여 높은 recall로 전체 text proposals을 생성한 다음 단어 분류를한다.
Gupta는 텍스트 감지에 Fully-Convolutional Regression Network를 학습하고 텍스트 인식에 단어 분류기를 사용한다.
Liao는 텍스트 감지에 SSD기반 방식을 사용하고 텍스트 인식에 CRNN을 사용한다.

최근 Li는 end-to-end text spotting방식을 제안하는데,
텍스트 감지를 위해 RPN에서 고안된 text proposal network를 사용하며 텍스트 인식을 위해 LSTM을 사용한다.

우리의 방식은 다른 방식과 비교하여 두 가지 이점이 있다.
1. 우리는 RoIRotate를 소개하며, 복잡하고 어려운 상황을 풀기위한 전혀다른 텍스트 감지 알고리즘을 사용한다.
반면 다른 방식들은 horizontal텍스트에만 적합하다.
2. 우리의 방식은 속도와 성능측에서 다른 방식보다 더 뛰어나며, 특히 cost-free한 인식 단계는 실시간 수행이 가능하게한다.
반면 다른 방식들은 600x800이미지 한장을 대략 900ms로 처리한다.
```

## 3. Methodology

```
FOTS는 자연 장면에서 모든 단어를 동시에 감지 및 인식을 하는 end-to-end framework이며,
shared convolutions, text detection branch, RoIRotate operation, text recognition branch로 구성된다.
```

### 3.1 Overall Architecture



## 4. Experiments
## 5. Conclusion
## 6. 공부 할 것들
