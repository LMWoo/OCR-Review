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
```
pipeline의 개략적인 개요는 
```

## 4. Experiments
## 5. Conclusion and Future Work
## 6. 공부 할 것들
```
candiate proposal
```
