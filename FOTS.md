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

<img width="584" alt="스크린샷 2021-06-19 오후 2 51 04" src="https://user-images.githubusercontent.com/80749934/122632464-e7e11500-d10d-11eb-8810-42c3900ec640.png">

```
위 그림은 shared network의 구조를 나타낸다.
text detection branch 및 recognition branch는 convolutional features를 공유한다.
shared network의 backbone은 ResNet-50이다.
FPN에서 영감을 받아, 우리는 low-level feature maps과 high-level feature maps을 연결한다.
shared convolutions에 의해 생성된 feature maps의 크기는 입력 이미지의 1/4크기이다.
text detection branch는 shared convolutions에 의해 생성된 features을 사용해 픽셀 별로 예측한다.
RoIRotate는 detection branch에 의해 생성된 text proposal을 이용해 features를 비율을 유지하여 정해진 높이로 변환한다.
text recognition branch는 region proposals안에 단어를 인식한다.
CNN과 LSTM은 시계열 정보를 인코딩하며, CTC로 디코딩된다.
아래 그림은 text recognition branch의 구조에 해당된다.
```

<img width="453" alt="스크린샷 2021-06-19 오후 3 05 16" src="https://user-images.githubusercontent.com/80749934/122632751-d0a32700-d10f-11eb-8a96-f3eb869463f6.png">

### 3.2 Text Detection Branch

```
텍스트 감지기로 fully convolutional network를 사용한다.
자연 장면 이미지에서 작은 텍스트가 많기 때문에, 
shared convolutions에서 입력이미지의 1/32에서 1/4의 크기의 feature maps으로 늘린다.
shared features를 추출하고, one convolution이 단어의 픽셀별 예측을 위해 공급된다.
첫 번째 채널은 각 픽셀별 확률을 계산한다.(Positive sample).
텍스트 영역에 픽셀들은 positive로 적용한다.
다음 4채널은 bonding box안에 픽셀에서 top, bottom, left, right까지의 거리를 예측한다.
마지막 채널은 bounding box의 방향을 예측한다.
최종 감지 결과는 positive samples에 thresholding과 NMS를 적용하여 생성된다.

이 실험에서, 펜스나 격자 같은 구별하기 힘든 텍스트 strokes와 유사한 패턴을 관찰한다.
이러한 패턴들을 잘 구별하기 위해 online hard example mining(OHEM)을 적용하며, 또한 OHEM은 클래스 불균형 문제를 해결한다.
이는 ICDAR2015데이터 셋에서 약 2%성능 개선을 보여준다.

detection branch의 loss function은 2가지 terms으로 구성된다.
text classification term과 bounding box regression term이다.
text classification term은 down-sample된 score map에 대한 픽셀별 loss로 볼 수 있다.
기존 텍스트 영역의 줄여진 버전만 positive area로 적용되며 이 사이의 공간은 "NOT CARE"로 적용된다.
```

```
아래 식은 classification에 대한 loss function이다.
```

<img width="506" alt="스크린샷 2021-06-19 오후 3 46 50" src="https://user-images.githubusercontent.com/80749934/122633841-9d639680-d115-11eb-9954-0a778730e91a.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?%5COmega)|the set of selected positive elements by OHEM in the score map|
|![](https://latex.codecogs.com/gif.latex?%7C%5Ccdot%7C)|the number of elements in a set|
|![](https://latex.codecogs.com/gif.latex?p_x)|the prediction of the score map|
|![](https://latex.codecogs.com/gif.latex?p%5E*_x)|the binary label that indicates text or non-text|

```
다음 식은 regression loss이며, 오브젝트 모양과 크기, 방향에 견고한 IoU loss와 rotation angle loss를 사용한다.
```

<img width="501" alt="스크린샷 2021-06-19 오후 4 00 19" src="https://user-images.githubusercontent.com/80749934/122634154-9473c480-d117-11eb-8b57-d28a81db8278.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?R_x)|the predicted bounding box|
|![](https://latex.codecogs.com/gif.latex?R%5E*_x)|the ground truth|
|![](https://latex.codecogs.com/gif.latex?IoU%28R_x%2C%20R%5E*_x%29)|the IoU loss between the predicted bounding box and the ground truth|
|![](https://latex.codecogs.com/gif.latex?%5Ctheta_x)|the predicted orientation|
|![](https://latex.codecogs.com/gif.latex?%5Ctheta%5E*_x)|the ground truth orientation|

```
최종 텍스트 감지 loss식은 다음과 같다. 
```

<img width="230" alt="스크린샷 2021-06-19 오후 4 17 42" src="https://user-images.githubusercontent.com/80749934/122634586-eddcf300-d119-11eb-8397-e024d31842ad.png">

### 3.3 RoIRotate(공부중)
### 3.4 Text Recognition Branch(공부중)

### 3.5 Implementation Details

```
ImageNet 데이터 셋에서 pre-trained model을 사용한다.
학습 과정은 두 단계가 있다.
먼저, Synth800k 데이터 셋으로 10epochs만큼 학습한 뒤,
실제 데이터로 file-tuning한다.
blurred text regions은 "DO NOT CARE"로 라벨링하며, 학습시 무시된다.

Data augmentation은 먼저 이미지의 긴쪽을 640에서 2560크기로 resize한다.
그 다음, [-10, 10](단위 degree) 범위에서 랜덤으로 회전시킨다.
그리고 이미지의 가로를 고정하고 높이를 0.8에서 1.2로 rescale한다.
마지막으로 이미지를 640x640만큼 자른다.

OHEM
각 이미지에 대해, 512 hard negative samples, 512 random negative samples, 모든 positive samples를 선택한다.
그 결과로, positive-to-negative비율은 1:60에서 1:3으로 증가된다.
그리고 bounding box회귀를 위해, 128 hard positive samples, 128 random positive samples을 선택한다.

Test
text detection branch에서 예측된 텍스트 영역을 얻은 후,
RoIRotate는 이 영역에서 thresholding과 NMS를 적용하고,
최종 인식 결과를 얻기위해 selected text features를 text recognition branch에 공급한다.
멀티 스케일 테스트는 모든 스케일의 결과가 합쳐지고 마지막 결과를 얻기위해 NMS로 다시 공급된다.
```


## 4. Experiments

### 4.1 Benchmark Datasets

#### 4.1.1 ICDAR 2015

```
이 데이터 셋은 1000 training images, 500 test images를 포함한다.
이러한 이미지는 Google glasses에 의해 위치에 관계없이 생성되며, 텍스트가 임의의 방향이 될 수 있다.
text spotting 테스트 단계에서 참고를 위한 3가지 lexicons이 있다. 
"Strong" lexicon은 이미지에 나타난 모든 단어를 포함해 100단어를 제공한다.
"Weak" lexicon은 전체 test set에 나타난 모든 단어를 포함한다.
"Generic" lexicondms 90k 단어 사전이다.

train
1. ICDAR 2017 MLT의 9000개 이미지를 사용하여 학습한다.
2. 1000개의 ICDAR 2015 train이미지와 229개의 ICDAR 2013 train이미지를 fine-tuning을 위해 사용한다.
```

#### 4.1.2 ICDAR 2017 MLT

```
이 데이터 셋은 9개의 다국어, 7200개의 training이미지, 1800개의 validation이미지, 9000개의 test images를 포함한다.
text spotting task가 없어 텍스트 감지 결과만 본다.
모델 학습을 위해 training set, validation set모두 사용한다.
```

#### 4.1.3 ICDAR 2013

```
이 데이터 셋은 229개의 training 이미지, 233개의 testing 이미지를 포함하며,
"Strong", "Weak", "Generic" lexicons을 제공한다.
다른 데이터 셋과 차이점은 horizontal text만 포함한다.
회전된 텍스트를 위해 설계된 방식은 이러한 텍스트에도 적합하다.
적은 training이미지 때문에 먼저 ICDAR 2017 MLT의 training, validation set를 사용하여 pre-trained model을 만들고,
ICDAR 2013의 training 이미지로 fine-tuning한다.
```

### 4.2 Comparison with Two-Stage Method

```
텍스트 감지 및 인식이 나눠진 방식과 다르게 연결된 방식은 각 작업에서 서로 이익을 얻을 수 있다.
이것을 증명하기 위해, 텍스트 감지와 인식을 개별적으로 훈련된 두 가지 시스템을 만든다.
detection network는 recognition branch가 제거되며 반대도 마찬가지이다.
결과적로 이렇게 나눠져 학습된 모델보다 동시에 학습된 모델의 성능이 훨씬 뛰어나다.

FOTS의 detection은 recognition의 label로 supervised learning을 수행하기 때문에 더 좋은 성능을 보여준다.
4가지 공통 이슈를 요약한다.
1. Miss : 텍스트 영역을 놓치는 것
2. False : 텍스트가 아닌 영역을 텍스트 영역으로 감지하는 것
3. Split : 하나의 텍스트 영역을 여러개의 영역으로 나누는 것
4. Merge : 여러개의 텍스트 영역을 하나의 영역으로 합치는 것
FOTS모델의 detection 모델은 text recognition label에 의한 supervised learning을 함으로써,
characters간의 미세한 정보까지 고려하여 학습하며 이는 비슷한 패턴을 가진 단어와 배경에 대해 강화된다.
따라서 위 4가지 이슈에 대해 강한 성능을 보여준다.
```

### 4.3 Comparisons with State-of-the-Art Results

```
모든 데이터 셋에서 큰 차이로 더 좋은 성능을 보여준다.
ICDAR 2017 MLT가 text spotting task가 없어, 감지 결과만 보여준다.
ICDAR 2013에 모든 텍스트 영역은 horizontal bounding box로 라벨링 되어있다.
FOTS model은 ICDAR 2017 MLT로 pre-train했으며 텍스트 영역의 방향을 예측할 수 있다.
ICDAR 2015에서 기존 방식보다 F-measure측에서 15%이상 성능향상을 보여준다.

단일 스케일 테스트를 위해 ICDAR2015, ICDAR 2017 MLT, ICDAR 2013에 대해 이미지의 긴 변을 2240, 1280, 920으로 resize한다.
멀티 스케일 테스트에 대해서는 3~5스케일을 적용한다.
```

### 4.4 Speed and Model Size

```
convolution sharing 전략을 통해 적은 계산과 메모리의 이익이 있으며,
"Our Two Stage"방식보다 2배정도 빨라, 실시간 속도를 유지할 수 있다.

ICDAR 2015및 ICDAR 2013에서 테스트 되었다.
이 데이터 셋은 68 text recognition labels이 있다.
모든 테스트 이미지를 검증하고 평균 속도를 계산한다.

ICDAR 2015의 이미지는 텍스트 감지에 대해 입력 크기로 2240x1260을 사용하며,
텍스트 인식을 위해 높이가 32로 맞춰진다.
ICDAR 2013의 이미지는 텍스트 감지에 긴 변을 920으로 resize하고 감지에 높이를 32로 맞춰 사용한다.
실시간 속도 달성을 위해, "FOTS RT"는 ResNet-50을 ResNet-34로 대체하며,
입력이미지를 1280x720으로 사용한다.
수정된 버전의 Caffe, TITAN-Xp GPU를 사용하여 테스트한다.
```

## 5. Conclusion

```
이 논문에서 oriented scene text spotting을 위한 end-to-end 학습 가능한 모델을 소개했다.(FOTS)
새로운 RoIRotate연산은 텍스트 감지와 인식을 end-to-end pipeline으로 통합한다.
convolutional features를 공유하여, 텍스트 인식 단계 비용이 거의 없고, 실시간 속도로 실행 되게 한다.
표준 벤치마크에 실험들은 제안된 방식이 효율성과 성능측에서 기존 방식보다 뛰어난 성능을 보여준다.
```

## 6. 공부 할 것들

```
RoIRotate
CRNN
LSTM
Attention
CTC Loss
```
