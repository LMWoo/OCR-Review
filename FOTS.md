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
## 3. Methodology
## 4. Experiments
## 5. Conclusion
## 6. 공부 할 것들
