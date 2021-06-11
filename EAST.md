# EAST: An Efficient and Accurate Scene Text Detector

# 논문
- https://arxiv.org/pdf/1704.03155v2.pdf

# 논문 요약
## 1. Introduction
## 2. Related Work
## 3. Methodology

```
제안된 알고리즘의 핵심 요소는 text instances의 존재와 전체 이미지에서의 geometries를 직접 학습되는 것이다.
이 모델은 각 픽셀별 words 또는 text lines의 예측을 출력한느 text 감지기에 적합한 fully-convolutional neural network이다.
이것은 candiate proposal, text region formation, word partition과 같은 중간 과정을 제거한다.
후처리 과정은 예측된 geometric shapes에 대해 thresholding과 NMS만을 포함한다.
이 감지기를 Efficient and Accuracy Scene Text detection pipeline(EAST)라고 명명한다.
```

### 3.1 Pipeline

<img width="650" alt="스크린샷 2021-06-11 오전 8 36 15" src="https://user-images.githubusercontent.com/80749934/121610083-2d5b6d80-ca90-11eb-9de8-7d3b50e65fa2.png">

```
pipeline의 개략적인 개요는 위 그림에 e로 나와있다.
이 알고리즘은 DenseBox의 일반적인 구조를 따르며,
이 구조는 FCN으로 한 이미지가 공급되며 픽셀별 text score map과 geometry map의 여러 채널이 생성된다.

예측된 것 채널 중 하나인 score map은 픽셀별로 [0,1]의 값을 가진다.
나머지 채널은 픽셀별 view로 부터 단어를 둘러싸는 geometries에 해당한다.
score는 같은 지점에 예측된 geometry shape의 신뢰도를 나타낸다.

rotated box(RBOX)및 quadrangle(QUAD)같은 text영역에 대한 두 geometry shape로 실험했다.
각 예측된 영역에 Thresholding을 적용한 다음,
predefined threshold보다 높은 geometries의 점수가 유효한 것으로 간주되며, non-maximum-suppression을 위해 저장된다.
NMS의 결과는 pipeline의 최종 출력으로 간주 된다.
```

### 3.2 Network Design

```

```

## 4. Experiments
## 5. Conclusion and Future Work
## 6. 공부 할 것들
```
candiate proposal
```
