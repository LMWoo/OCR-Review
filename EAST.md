# EAST: An Efficient and Accurate Scene Text Detector

# 논문
- https://arxiv.org/pdf/1704.03155v2.pdf

# 논문 요약
## 1. Introduction

1. 기존방식은 여러 단계 및 요소로 구성되어 최적화 되지 않고 시간이 많이 걸림
2. 따라서, 2단계의 텍스트 감지 pipeline을 제안함
3. 첫 번째는 직접 word또는 text-line예측을 하는 fully convolutional network(FCN)모델을 이용
4. 두 번째는 예측된 word또는 text-line을 Non-Max Suppression으로 보내 최종 결과를 산출
5. 제안된 알고리즘은 정확성과 속도에서 이전 최점단 방식보다 뛰어난 성능을 달성함

## 2. Related Work

*딥러닝 이전*

1. SWT, MSER - edge감지 또는 extremal region extraction을 통해 character후보 찾음
2. Zhang 등은 텍스트의 대칭성과 영역 감지를 위한 다양한 features를 이용함
3. FASText는 Stroke extraction을 위한 FAST감지기를 개조 및 수정한 fast text dectection system

그러나 이러한 방법은 특히, 낮은 해상도 및 기하 왜곡과 같은 상황을 다룰때 딥러닝 방식보다 뒤쳐진다.

*딥러닝*


1. Huang 등은 MSER로 text 후보를 찾고 false positives를 줄이기 위해 a deep convolutional network을 사용함
2. Jaderberg 등은 sliding-window방식으로 이미지를 스캔하고 convolutional neural network로 각 크기에 대해 heatmap을 생성한 후,
word후보를 찾기 위해 CNN과 ACF를 사용하고 regression으로 불순물을 제거함
3. Tian 등은 수직 anchors를 발전시키고, 수평의 text lines를 감지하기 위해 CNN-RNN모델을 구성함
4. Zhang 등은 heatmap생성을 위해 FCN을 이용하고 방향 추적을 위해 component projection를 이용함

```
이러한 방식들은 표준 벤치마크에서 우수한 성능을 얻었지만,
post filtering에 의한 false positive제거, candidate aggregation, line formation, word partition으로 구성된다.
이러한 복잡한 단계와 요소는 exhaustive tuning을 요구하며 최적화 되지않은 성능을 이끌고 전체 pipeline의 처리 시간이 늘어난다.

이 논문에서는 텍스트 감지의 최종 목표를 직접 타겟으로하는 FCN기반 pipeline을 고안한다.
이 모델은 불필요한 중간 단계 및 요소를 없애며, end-to-end 방식을 가능할 수 있게 한다.
최종 신경망 네트워크는 성능과 속도에서 이전 방식들을 능가한다. 
```

## 3. Methodology

```
제안된 알고리즘의 핵심 요소는 전체 이미지에서 텍스트의 존재와 기하 요소를 직접적으로 예측하기 위해 훈련된 신경망 모델이다.
이 모델은 각 픽셀별 words 또는 text lines의 예측을 출력하는 텍스트 감지에 적합한 fully-convolutional neural network이다.
이 것은 candiate proposal, text region formation, word partition과 같은 중간 과정을 제거한다.
후처리 단계는 예측된 기하학적 모양에 대해 오직 thresholding과 NMS만을 포함한다.
이 감지기를 Efficient and Accuracy Scene Text detection pipeline(EAST)라고 명명한다.
```

### 3.1 Pipeline

<img width="650" alt="스크린샷 2021-06-11 오전 8 36 15" src="https://user-images.githubusercontent.com/80749934/121610083-2d5b6d80-ca90-11eb-9de8-7d3b50e65fa2.png">

1. pipeline의 개략적인 개요(그림 e)
2. DenseBox의 구조를 따르며 FCN으로 이미지가 입력되고, score map및 geometry map이 출력됨
3. geometry map은 단어를 둘러싸는 기하 요소에 해당
4. score는 같은 지점에 예측된 geometry shapes의 신뢰도를 나타냄, [0,1]의 값을 가짐
5. 텍스트 영역에 대한 rotated box(RBOX)및 quadrangle(QUAD)에 대해 각각의 geometry로 실험 및 loss 함수 설계
6. 미리 지정한 threshold보다 높은 score, geometries을 남기고, 이를 NMS을 통해 최종 결과를 출력

### 3.2 Network Design

```
text감지를 위한 신경망 설계를 할 때, 다음과 같은 요소를 고려해야한다.
```

1. 단어의 크기가 매우 다양함
2. 큰 크기의 단어는 신경망 나중 단계의 features가 필요함
3. 반대로 작은 크기의 단어는 신경망 초기 단계의 features가 필요함
4. 2, 3번의 이유로 다른 levels의 features를 사용해야함

```
HyperNet은 위 조건을 만족하지만 커다란 feature maps에 많은 채널을 병합해 계산량이 크게 증가한다.
따라서, upsampling branch를 작게 유지하면서 feature maps을 병합하는 U-shape를 채택했다.
결과적으로 다른 levels의 features를 활용하고 계산량이 적은 신경망을 갖게 된다.
```

<img width="406" alt="스크린샷 2021-06-11 오전 9 25 23" src="https://user-images.githubusercontent.com/80749934/121613375-41569d80-ca97-11eb-8767-6a79f618fbd8.png">

```
위 그림은 모델 계략도이며, 세 가지 부분으로 나눠진다 : feature extractor(stem), feature-merging(branch), output layer
```

*feature extractor(stem)*

1. ImageNet에서 사전 훈련되며, convolution및 pooling레이어를 가진 신경망
2. stem에서 4가지의 features maps이 나오며, 각 크기는 입력 이미지의 1/32, 1/16, 1/8, 1/4이다.
3. 실험에서 PVANet및 VGG16모델을 채택했으며, pooling-2에서 pooling-5까지의 feature maps이 얻어진다.

*feature-merging(branch)*

```
다음 식들은 branch관련 공식을 나타내며, 각 merging stage는 다음 단계로 요약된다.
```

<img width="400" alt="스크린샷 2021-06-11 오전 9 57 32" src="https://user-images.githubusercontent.com/80749934/121615293-7a910c80-ca9b-11eb-8779-3e379b252042.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?g_i)|the merge base|
|![](https://latex.codecogs.com/gif.latex?h_i)|the merged feature map|
|operator [·;·]|concatenation along the channel axis|

1. 현재 feature map크기가 2배가 되는 unpooling layer로 공급되고, 전 단계 feature map과 concatenate된다.
2. conv1x1 bottleneck으로 channel의 수를 줄여 계산량을 줄인다.
3. 정보를 융합하는 conv3x3 layer는 이 merging단계의 출력을 생산한다.
4. 마지막 branch의 conv3x3 layer은 feature map을 output layer로 공급한다.

*output layer*

1. score map : (conv1x1, 1 channel)연산으로 score map을 출력
2. geometry map : (conv1x1, multi channel)연산으로 geometry map을 출력, RBOX 또는 QUAD가 됨
3. RBOX : pixel위치와 사각형의 boundaries인 left, top, right, bottom의 거리(4 channels), rotation angle(1 channel)
4. QUAD : pixel위치와 각 vertices의 좌표 변화량(8 channels)

### 3.3 Label Generation
#### 3.3.1 Score Map Generation for Quadrangle

<img width="536" alt="스크린샷 2021-06-11 오전 11 00 53" src="https://user-images.githubusercontent.com/80749934/121619941-60a7f780-caa4-11eb-9fcc-5a38a144380d.png">

score map의 positive영역은 원래의 영역보다 줄여저 설계된다.(Table 1.의 a)

사각형 Q에 대해 ![](https://latex.codecogs.com/gif.latex?p_i) = {![](https://latex.codecogs.com/gif.latex?x_i), ![](https://latex.codecogs.com/gif.latex?y_i)}는 시계방향의 각 사각형 정점이다.
Q를 줄이기 위해, 각 ![](https://latex.codecogs.com/gif.latex?p_i)에 대해 reference length(![](https://latex.codecogs.com/gif.latex?r_i))를 구해야한다.

![](https://latex.codecogs.com/gif.latex?D%28p_i%2C%20p_j%29)는 ![](https://latex.codecogs.com/gif.latex?p_i%2C%20p_j)사이의 거리이다.

<img width="320" alt="스크린샷 2021-06-11 오전 11 14 34" src="https://user-images.githubusercontent.com/80749934/121620975-3eaf7480-caa6-11eb-8cf9-07b67961cf5d.png">

1. 마주보는 변의 2개의 쌍에 대해 평균 길이가 긴 것을 'longer'로 결정한다.
2. 각 변에 대해 ![](https://latex.codecogs.com/gif.latex?0.3r_i%2C%200.3r_%7B%28i%7Emod%7E4%29%20&plus;%201%7D)만큼 안쪽으로 끝점을 이동시켜 줄인다.

#### 3.3.2 Geometry Map Generation

"RBOX"

1. 최소한 영역으로 region을 커버하는 회전된 직사각형을 생성한다.
2. positive score를 가진 각 픽셀에 대해, text box의 4 boundaries까지의 거리를 계산
3. RBOX의 4 channels ground truth에 그 값을 넣는다.

"QUAD"

1. positive score를 가진 각 픽셀에 대해 사각형 4개의 정점으로부터 좌표 변화량을 나타낸다.

### 3.4 Loss Functions

<img width="150" alt="스크린샷 2021-06-11 오전 11 43 30" src="https://user-images.githubusercontent.com/80749934/121623250-4ffa8000-caaa-11eb-8e6b-3c5f368add93.png">

```
위 식은 최종 Loss공식이다.
이 식에서 Ls와 Lg는 각각 score map과 geometry에 대한 loss를 나타낸다.
![](https://latex.codecogs.com/gif.latex?%5Clambda_g)는 이 2개의 loss에 대한 가중치이며, 실제 실험에는 1로 세팅한다.
```

#### 3.4.1 Loss for Score Map

```
대부분 최첨단 감지 pipelines에서 학습 이미지는 target objects의 분균형한 분포를 해결하기 위해,
balanced sampling과 hard negative mining으로 처리된다.
그러면 성능을 개선하나, 그것은 복잡한 절차, 더 많은 paramters, 복잡한 pipeline을 도입한다.
따라서, 더 간단한 학습을 위해 class-balanced cross-entropy를 사용한다.
```

<img width="456" alt="스크린샷 2021-06-11 오후 12 08 38" src="https://user-images.githubusercontent.com/80749934/121625172-ec725180-caad-11eb-92fb-b7a06ab4807a.png">

<img width="229" alt="스크린샷 2021-06-11 오후 12 11 02" src="https://user-images.githubusercontent.com/80749934/121625302-293e4880-caae-11eb-8308-bee39cda655c.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D) ( = ![](https://latex.codecogs.com/gif.latex?F_s))|the prediction of the score map|
|![](https://latex.codecogs.com/gif.latex?Y%5E*)|the ground truth|
|![](https://latex.codecogs.com/gif.latex?%5Cbeta)|the balancing factor between positive and negative samples|

#### 3.4.2 Loss for Geometries

```
텍스트 감지에 한 가지 문제는 텍스트 크기가 natural scene images에서 매우 다양하다는 것이다.
L1 또는 L2 loss를 직접 사용하는 것은 loss bias를 더 크고 긴 text regions으로 이끈다.
크고 작은 text region에 대한 정확한 text geometry 예측을 생성하기 위해, regression loss는 scale-invariant이어야한다.
그러므로, RBox regression의 AABB에서 IoU loss를 적용하고,
QUAD regression에 대해 scale-normalize된 smoothed-L1을 사용한다.
```

*RBOX*

```
AABB part를 위해, 다른 크기의 objects에 대해 불변이므로 IoU loss를 채택한다.
다음 식은 IoU loss를 나타낸다.
```

이 식에서 ![](https://latex.codecogs.com/gif.latex?%5Chat%7BR%7D)는 predicted AABB geometry, ![](https://latex.codecogs.com/gif.latex?R%5E*)는 ground truth를 나타낸다.

<img width="428" alt="스크린샷 2021-06-11 오후 1 28 53" src="https://user-images.githubusercontent.com/80749934/121631030-4a586680-cab9-11eb-97aa-64d9fe1bcae0.png">

```
다음 식은 교집합 직사각형의 가로와 세로를 나타낸다.
d1, d2, d3, d4는 픽셀에서 각각 사각형의 left, top, right, bottom까지의 거리를 나타낸다. 
```

<img width="300" alt="스크린샷 2021-06-11 오후 1 29 17" src="https://user-images.githubusercontent.com/80749934/121631039-4debed80-cab9-11eb-8d29-23f675bf690a.png">

```
다음 식은 직사각형의 합집합 나타낸다.
결과적으로 위, 아래 식으로 intersection / union area는 쉽게 계산된다.
```

<img width="338" alt="스크린샷 2021-06-11 오후 1 29 40" src="https://user-images.githubusercontent.com/80749934/121631040-4e848400-cab9-11eb-97ef-ddab15e7d4ca.png">

```
다음 식은 각도에 대한 loss이다.
```

이 식에서 ![](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Ctheta%7D)은 예측된 angle이고 ![](https://latex.codecogs.com/gif.latex?%5Ctheta%5E*)은 ground truth이다.

<img width="266" alt="스크린샷 2021-06-11 오후 1 29 50" src="https://user-images.githubusercontent.com/80749934/121631042-4f1d1a80-cab9-11eb-81b9-f5a49ff5d821.png">

```
다음 식은 최종 geometry loss이며 AABB loss와 angle loss의 가중치 합이다.
```

이 실험에서 ![](https://latex.codecogs.com/gif.latex?%5Clambda_%5Ctheta)는 10으로 세팅한다.

<img width="209" alt="스크린샷 2021-06-11 오후 1 30 00" src="https://user-images.githubusercontent.com/80749934/121631046-4fb5b100-cab9-11eb-9bdf-0fd2939dc113.png">

```
AABB의 Loss를 회전 각에 관계없이 계산하였다.
이 것은 각도가 완벽히 예측 될 때, 사각형 IoU의 근사처럼 보여 질 수 있다.
훈련 중에 그런 경우는 없지만, R을 예측하는 것을 학습하도록 올바를 기울기를 부과 할 수 있다.
```

*QUAD*

```
한 방향으로 더 긴 단어의 사각형을 위해 설계된 extra normalization term이 추가하여 smoothed-L1 loss를 확장한다.
Q의 모든 좌표 값을 다음과 같은 순서대로 설정한다.
```

<img width="310" alt="스크린샷 2021-06-11 오후 2 32 51" src="https://user-images.githubusercontent.com/80749934/121635862-f2722d80-cac1-11eb-984b-bab5c248c235.png">

그런 다음 Loss는 다음과 같다.

<img width="392" alt="스크린샷 2021-06-11 오후 2 34 01" src="https://user-images.githubusercontent.com/80749934/121635968-277e8000-cac2-11eb-87c8-b54267a7461a.png">

다음 식에 의해, 정규화 term ![](https://latex.codecogs.com/gif.latex?N_%7BQ%5E*%7D)는 사각형 변중 가장 짧은 변이다.
그리고 ![](https://latex.codecogs.com/gif.latex?P_Q)는 다른 정점 순서를 가진 ![](https://latex.codecogs.com/gif.latex?Q%5E*)의 같은 사각형의 집합이다. training데이터 셋에서 사각형 annotation이 일치하지 않아 순서 변경이 필요하다.

<img width="292" alt="스크린샷 2021-06-11 오후 2 41 45" src="https://user-images.githubusercontent.com/80749934/121636662-331e7680-cac3-11eb-9da3-ea9a9a203b3a.png">

### 3.5 Training

```
이 network는 ADAM optimizer를 이용해 학습된다.
ADAM의 학습률은 1e-3에서 시작하여 27300 batch마다 1/10씩 줄어 1e-5에서 멈춘다.
학습 속도를 높이기 위해, 이미지에서 균일하게 512x512크기로 샘플링하여 24크기의 미니 배치를 형성한다.
이 network는 성능 개선이 멈출 때 까지 계속 train된다.
```

### 3.6 Locality-Aware NMS

```
최종 결과를 위해, thresholding후 남은 geometries가 NMS에 의해 병합되야한다.
단순 NMS알고리즘은 geometries후보의 수 n에 O(n^2)의 속도로 실행되며
예측으로 부터 수만 개의 geometries를 마주하기 때문에 받아들일 수 없다.

가까운 픽셀으로 부터 geometries가 서로 연관 된다는 하에 geometries를 row by row로 병합할 것을 제안한다.
geometries를 병합하는 동안 현재 geometry가 마지막 병합된 것과 반복적으로 병합될 것이다.
이 개선된 기술은 가장 빠를때 O(n)로 실행되며 최악의 경우 기존 알고리즘 속도와 같다.
```

<img width="400" alt="스크린샷 2021-06-12 오전 10 10 25" src="https://user-images.githubusercontent.com/80749934/121760654-dd030f00-cb66-11eb-887a-009f3ccf9afd.png">

WEIGHTEDMERGE(g, p)에서, 병합된 사각형의 좌표는 두 개의 사각형의 score에 의해 가중치 평균이된다.
구체적으로, a = WEIGHTEDMERGE(g, p)이면,
![](https://latex.codecogs.com/gif.latex?a_i%20%3D%20V%28g%29g_i%20&plus;%20V%28p%29p_i) 및 ![](https://latex.codecogs.com/gif.latex?V%28a%29%20%3D%20V%28g%29%20&plus;%20V%28p%29),
여기서 ![](https://latex.codecogs.com/gif.latex?a_i)는 a의 i번 좌표이며, V(a)는 a의 geometry점수이다.

## 4. Experiments

제안된 알고리즘과 기존 방식과 비교를 위해, 3개의 데이터셋(ICDAR2015, COCO-Text, MSRA-TD500)에 대해 질적 및 정량적 실험을 수행한다.

### 4.1 Benchmark Datasets

*ICDAR2015*

1. 1000개의 training images, 500개의 test images
2. 텍스트 영역은 4 vertices of the quadrangle(논문에서 QUAD와 대응)
3. 최소 면적을 가진 rotated rectangle(RBOX)를 생성
4. Google Glass에서 촬영했으며, 텍스트가 임의의 방향, motion blur, 저 해상도의 문제가 있음
5. ICDAR2013의 229개의 training images또한 사용

*COCO-Text*

1. 가장 큰 데이터 셋.
2. MS-COCO의 이미지를 재사용.
3. 총 63686개의 이미지가 있으며, 43686개의 training set, 20000개의 test set으로 선택
4. 텍스트 영역은 axis-aligned bounding box(AABB)형태이고, angle은 0으로 세팅
5. ICDAR2015와 같은 data processing 방식

*MSRA-TD500*

1. 300개의 training images, 200개의 test images
2. 텍스트 영역은 임의의 방향과 문장 level, RBOX형태로 annoate됨
3. English, Chinense를 포함
4. training images가 적어, HUST-TR400의 400개의 이미지를 이용

### 4.2 Base Networks

```
COCO-Text를 제외하고, 모든 text detection 데이터 셋은 object detection데이터 셋에 비해 작다.
그러므로, 단일 network를 채택하면, overfitting또는 under-fitting때문에 어려움을 겪을 수 있다.
제안된 프레임워크를 평가하기 위해, 모든 데이터 셋에서 다른 출력의 기하 구조를 가진 3가지 base networks를 사용한다.
```

*VGG16*

1. text detection포함 fine-tuning을 지원하기 위해 base network로 널리 사용됨
2. 수용 영역이 적고, 다소 큰 네트워크라는 두 가지 단점이 있음

*PVANET*

1. Faster-RCNN framework의 feature extractor의 대체를 목표함
2. 너무 작아, 충분히 GPU 병렬 연산이 가능
3. 기존 PVANET 채널의 2배인 PVANET2x를 채택함
4. PVANET2x는 기존 보다 약간 느리지만 더 많은 병렬 처리 연산을 수행함
5. 마지막 출력의 수용영역은 809이며, VGG16보다 훨씬 크다.

### 4.3 Qualitative Results

```
제안된 알고리즘은 불균일한 조명, 낮은 해상도, 다양한 방향, 투영으로 인한 왜곡과 같은 어려운 상황을 다룰 수 있다.
게다가, NMS에 voting mechanism 때문에, 다양한 모양의 텍스트를 가진 videos에서 높은 안정성을 보여준다.

훈련된 모델은 매우 정확한 geometry maps과 score map을 생산하고,
다양한 방향의 텍스트 감지를 쉽게 할 수 있다.
```

### 4.4 Quantitative Results

제안된 알고리즘은 ICDAR2015및 COCO-Text에서 큰 차이로 이전의 최첨단 방식보다 뛰어남

*ICDAR2015*

1. 최고 성능이 나온 알고리끼리 비교하면 F-score가 이전 보다 0.16만큼 높음
2. VGG를 사용했을때는 QUAD출력 일 때 0.0924, RBOX출력 일 때 0.116만큼 높음

*COCO-Text*

1. F-score에서 0.0614, Recall에서 0.053넘는 개선
2. COCO-Text가 크고 까다로운 벤치마크임을 고려하면, 제안된 알고리즘의 장점이 확인됨

이전 방법에 비해 제안된 알고리즘은 최종 목표를 직접 타겟으로하고 중복 프로세스를 제거하는 간단한 텍스트 감지 파이프라인이,
복잡한, 대규모 신경망 모델보다 성능이 능가할 수 있음을 증명한다.

*MSRA-TD500*

1. 최고 성능(Ours+PVANET2x)에서 이전 보다 F-score는 0.0208, precision은 0.0428만큼 개선
2. VGG16의 수용 영역이 PVANET및 PVANET2x보다 작아 PVANET및 PVANET2x성능이 좋음
3. MSRA-TD500의 검증 프로토콜은 word level대신 line level 텍스트 감지 알고리즘 출력을 

*ICDAR2013*

1. 최고 성능(Ours+PVANET2x)에서 recall, precision, F-score가 0.8267, 0.9264, 0.8737을 달성

### 4.5 Speed Comparision

```
ICDAR2015 데이터셋에 500개 test images를 원래 해상도로 실행함
단일 NVIDIA Titan X그래픽 타드 및 Intel E5-2670 v3(2.30 GHz)CPU로 실험
제안된 방법의 후처리는 thresholding및 NMS를 포함, 다른 방식은 원래의 논문을 참고

제안된 알고리즘은 이전 최첨단 알고리즘보다 훨씬 뛰어나며, 간단하고 효율적인 pipeline으로 계산 비용이 훨씬적다.
가장 빠른 방식(Ours+PVANET)은 16.8FPS로 실행, 가장 느린 방식(Ours+VGG16)은 6.52FPS로 실행됨
최고 성능(Ours+PVANET2x)는 13.2FPS로 실행됨
```

### 4.6 Limitations

```
탐지기가 다룰 수 있는 텍스트의 최대 크기는 수용영역에 비례한다.
이는 이미지에 text lines처럼 긴 텍스트 영역을 예측하는데 제한된다.
또한, ICDAR2015데이터 셋은 작은 영역의 텍스트만 가지므로 수직 텍스트에 대해 부정확한 예측을 할 수 있다.
```

## 5. Conclusion and Future Work

적절한 손실 함수를 통합함으로써, 감지기는 텍스트 영역에 대해 rotated rectangles또는 quadrangles를 예측할 수 있다.
향후 연구 방향
1. curved text를 직접 감지하도록 geometry 공식을 조정
2. detector와 recognition 통합
3. 아이디어를 일반적인 물체 감지로 확장

## 6. 공부 할 것들

```
extremal region extraction
exhaustive tuning
hard negative mining
smoothed-L1 loss
extra normalization term
Locality-Aware NMS
voting mechanism
class-balanced cross-entropy
candiate proposal
```
