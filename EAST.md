# EAST: An Efficient and Accurate Scene Text Detector

# 논문
- https://arxiv.org/pdf/1704.03155v2.pdf

# 논문 요약
## 1. Introduction
## 2. Related Work
## 3. Methodology

제안된 알고리즘의 핵심 요소는 text instances의 존재와 전체 이미지에서의 geometries를 직접 학습되는 것이다.
이 모델은 각 픽셀별 words 또는 text lines의 예측을 출력한느 text 감지기에 적합한 fully-convolutional neural network이다.
이것은 candiate proposal, text region formation, word partition과 같은 중간 과정을 제거한다.
후처리 과정은 예측된 geometric shapes에 대해 thresholding과 NMS만을 포함한다.
이 감지기를 Efficient and Accuracy Scene Text detection pipeline(EAST)라고 명명한다.

### 3.1 Pipeline

<img width="650" alt="스크린샷 2021-06-11 오전 8 36 15" src="https://user-images.githubusercontent.com/80749934/121610083-2d5b6d80-ca90-11eb-9de8-7d3b50e65fa2.png">

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

### 3.2 Network Design

text감지를 위해 neural networks를 설계할 때, 몇 가지 요소를 고려해야한다.
단어 영역의 크기는 매우 다양해서, 큰 크기의 단어의 존재를 결정하는 것은 neural network의 나중 단계에서 features을 필요로 하고,
반대로, 작은 크기의 단어를 둘러싸는 정확한 geometry를 예측하는 것은 초기 단계에서 low-level정보를 필요로한다.
그러므로, 이러한 요구를 충족하기 위해서 다른 levels의 features를 사용해야한다.
HyperNet은 feature maps에 대해 이러한 조건을 만족한다.
하지만 커다란 feature maps에서 많은 채널을 병합하는 것은 나중 단계이서 계산 오버헤드가 크게 증가할 것이다.

이를 해결하기 위해, upsampling branch를 작게 유지하면서 feature maps을 병합하는 U-shape의 아이디어를 채택했다.
결국 다른 levels의 features를 활용하고 계산 비용을 적게유지할 수 있는 network를 갖게 된다.

<img width="406" alt="스크린샷 2021-06-11 오전 9 25 23" src="https://user-images.githubusercontent.com/80749934/121613375-41569d80-ca97-11eb-8767-6a79f618fbd8.png">

위 그림은 모델의 계략도를 나타낸다.
이 모델은 세 가지 부분으로 나눠진다 - feature extractor(stem), feature-merging(branch), output layer

stem은 ImageNet데이터 셋에서 사전훈련된 convolutional network일 수 있다.
feature maps의 4가지 levels은 stem에서 얻고, 각 크기는 입력 이미지의 1/32, 1/16, 1/8, 1/4이다.
위 그림은 PVANet을 나타낸다.
실험에서, 잘 알려진 VGG16모델을 채택했으며, pooling-2에서 pooling-5까지의 feature maps이 얻어진다.

<img width="400" alt="스크린샷 2021-06-11 오전 9 57 32" src="https://user-images.githubusercontent.com/80749934/121615293-7a910c80-ca9b-11eb-8779-3e379b252042.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?g_i)|the merge base|
|![](https://latex.codecogs.com/gif.latex?h_i)|the merged feature map|
|operator [·;·]|concatenation along the channel axis|

위 그림과 식들은 feature-merging branch관련 공식을 나타낸다.
각 merging stage는 다음 단계로 요약된다.
1. 마지막 단계의 feature map크기를 2배로 늘리는 unpooling layer로 공급되고, 현재 feature map과 concatenate된다.
2. conv1x1 bottleneck은 channel의 수를 줄여 계산량을 줄인다.
3. 정보를 융합하는 conv3x3 layer는 이 merging단계의 출력을 생산한다.
4. 마지막 단계의 conv3x3 layer는 merging branch의 마지막 feature map을 생산하며, output layer에 이를 공급한다.

branch단계에서 convolutions에 대한 channels의 수를 작게 유지하며, 
stem에 대한 계산 overhead의 극히 일부만 추가하여 효율적인 network계산을 한다.
마지막 output layer는 feature map의 32개 channels을 score map의 1 channel, 
geometry map의 multi-channel channel로 산출하는 conv1x1연산자를 포함한다.
geometry출력은 RBOX또는 QUAD중 하나가 될 수 있다.

RBOX는 axis-aligned bounding box(AABB)R의 4 channels, rotation angle의 1 channel로 나타낸다.
R은 pixel 위치에서 각각 사각형의 boundaries인 left, top, right, bottom의 거리를 나타낸다.

QUAD는 pixel 위치에서 각 vertices의 좌표 변화를 나타내며, geometry output은 8 channels을 포함한다.

### 3.3 Label Generation
#### 3.3.1 Score Map Generation for Quadrangle

<img width="536" alt="스크린샷 2021-06-11 오전 11 00 53" src="https://user-images.githubusercontent.com/80749934/121619941-60a7f780-caa4-11eb-9fcc-5a38a144380d.png">

score map의 positive영역은 원래의 영역보다 줄여저 설계된다.(Table 1.의 a)

사각형 Q에 대해 ![](https://latex.codecogs.com/gif.latex?p_i) = {![](https://latex.codecogs.com/gif.latex?x_i), ![](https://latex.codecogs.com/gif.latex?y_i)}는 시계방향의 각 사각형 정점이다.
Q를 줄이기 위해, 각 ![](https://latex.codecogs.com/gif.latex?p_i)에 대해 reference length(![](https://latex.codecogs.com/gif.latex?r_i))를 구해야한다.

![](https://latex.codecogs.com/gif.latex?D%28p_i%2C%20p_j%29)는 ![](https://latex.codecogs.com/gif.latex?p_i%2C%20p_j)사이의 거리이다.

<img width="320" alt="스크린샷 2021-06-11 오전 11 14 34" src="https://user-images.githubusercontent.com/80749934/121620975-3eaf7480-caa6-11eb-8cf9-07b67961cf5d.png">

1. 마주보는 변의 2개의 쌍에 대해 평균 길이가 긴 것을 'longer'로 결정한다.
2. 각 변에 대해 0.3![](https://latex.codecogs.com/gif.latex?r_i), 0.3![](https://latex.codecogs.com/gif.latex?r_%7B%28i%7Emod%7E4%29&plus;1%7D)만큼 안쪽으로 끝점을 이동시켜 줄인다.

#### 3.3.2 Geometry Map Generation

QUAD style로 데이터 셋의 text regions을 annoate하기 위해, 
먼저 최소한 영역으로 region을 커버하는 회전된 직사각형을 생성한다.
그런 다음, positive score를 가진 각 픽셀에 대해, text box의 4 boundaries까지의 거리를 계산하고,
RBOX의 4 channels ground truth에 그 값을 넣는다.
QUAD ground truth에 대해, 8-channel geometry map에서 positive score의 각 픽셀에 대한 값은,
사각형의 4개의 정점으로부터 좌표 변화량이다.

### 3.4 Loss Functions

<img width="150" alt="스크린샷 2021-06-11 오전 11 43 30" src="https://user-images.githubusercontent.com/80749934/121623250-4ffa8000-caaa-11eb-8e6b-3c5f368add93.png">

Ls와 Lg는 각각 score map과 geometry에 대한 loss를 나타내며, ![](https://latex.codecogs.com/gif.latex?%5Clambda_g)는 이 2개의 loss에 대한 가중치이다. 실제 실험에는 1로 세팅한다.

#### 3.4.1 Loss for Score Map

대부분 최첨단 감지 pipelines에서 학습 이미지는 target objects의 분균형한 분포를 해결하기 위해,
balanced sampling과 hard negative mining으로 신중하게 처리된다.
그렇게 하면 잠재적으로 network성능을 개선할 수 있다.
그러나, 그러한 기술을 이용하는 것은 반드시 구별할 수 없는 단계, tuning을 위한 더 많은 parameters 더 복잡한 pipeline을 도입한다.

더 간단한 training절차를 수월하게 하기 위해, class-balanced cross-entropy를 사용한다.

<img width="456" alt="스크린샷 2021-06-11 오후 12 08 38" src="https://user-images.githubusercontent.com/80749934/121625172-ec725180-caad-11eb-92fb-b7a06ab4807a.png">

<img width="229" alt="스크린샷 2021-06-11 오후 12 11 02" src="https://user-images.githubusercontent.com/80749934/121625302-293e4880-caae-11eb-8308-bee39cda655c.png">

|기호|정의|
|----|----|
|![](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D) ( = ![](https://latex.codecogs.com/gif.latex?F_s))|the prediction of the score map|
|![](https://latex.codecogs.com/gif.latex?Y%5E*)|the ground truth|
|![](https://latex.codecogs.com/gif.latex?%5Cbeta)|the balancing factor between positive and negative samples|

#### 3.4.2 Loss for Geometries

text감지에 한 가지 문제는 text크기가 natural scene images에서 매우 다양하다는 것이다.
L1 또는 L2 loss를 직접 사용하는 것은 loss bias를 더 크고 긴 text regions으로 이끈다.
크고 작은 text region에 대해 정확한 text geometry 예측을 생성하기 위해, regression loss는 scale-invariant이어야한다.
그러므로, RBox regression의 AABB에 IoU loss를 적용하고,
QUAD regression에 대해 scale-normalize된 smoothed-L1을 사용한다.

RBOX
AABB part를 위해, 다른 크기의 objects에 대해 불변이므로 IoU loss를 채택한다.


## 4. Experiments
## 5. Conclusion and Future Work
## 6. 공부 할 것들
class-balanced cross-entropy
candiate proposal
