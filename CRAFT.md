## CRAFT

### 논문
- https://arxiv.org/pdf/1904.01941.pdf

### 논문 요약
1. Introduction
```
딥러닝 기반 텍스트 감지기 대부분 word-level bounding boxes를 localize하는데 집중하고 있다.
그러나, 이런 방식은 다양한 모양의 texts를 a single bounding box로 감지하기 힘들다.
그 대신, character-level감지는 bottom up방식으로 characters를 연결하여 이러한 어려움에 이점이 있다.
불행하게도, 대부분 데이터 셋들이 character-level annotations을 제공하지 않는다.
그리고 character-level ground truths를 얻기에 쉽지 않다.

이 논문에서는 characters를 localization하고 이들을 연결하는 새로운 text 감지기를 제안한다.
이를 Character Region Awareness For Text detection(CRAFT)라고 칭한다.
CRAFT는 character region score와 affinity score를 생성하도록 설계되었다.
region score는 각 characters를 localize하고 affinity score는 이들을 하나의 instance로 그룹화한다.
character-level annotation이 부족하므로,
word-level datasets에서 character-level ground truths를 추정하는 weakly-supervised learning방식을 제안한다.
이 모델은 다양한 모양을 포함한 데이터 셋에서 최첨단 성능을 보여준다.
```

2. Related Work
```
딥러닝 이전text detection은 주로 hand-crafted features방식이었다.(MSER, SWT)
최근에, 딥러닝 기반 텍스트 감지기는 object detection방식을 사용하고 있다.(SSD, Faster R-CNN, FCN)
```

*Regression-based text detectors*
```
다양한 box regression방식의 텍스트 감지기는 object detection방식으로 채택된다.
objects와 달리 texts는 불규칙한 모양을 가지는 문제점이있다.

TextBoxes - texts의 다양한 모양을 감지하기위해 convolutional kernels과 anchor boxes를 수정했다.
DMPNet - quadrilateral sliding windows를 통합하여 이 문제를 줄였다.
Rotation-Sensitive Regression Detector(RSDD) - convolutional filters를 회전시켜 이를 해결했으나,
모든 모양의 texts를 감지하는데 구조상의 제한이 있다.
```

*Segmentation-based text detectors*
```
segmentation방식은 pixel-level에서 text regions을 찾는 방식이다.
Multi-scale FCN, Holistic-prediction, PixelLink는 segmentation을 기반으로 text를 감지한다.

SSTD - Attention mechanism을 이용해 regression과 segmentation에서 이익을 얻었다.
TextSnake - text region과 center line 예측하여 text를 감지했다.
```

*End-to-end text detectors*
```
end-to-end방식은 recognition결과를 활용하여 정확도를 높이기 위해 detection과 recognition을 동시에 하는 방식이다.
이는 배경과 비슷한 텍스트에 대해 더욱 견고하게 한다.

FOTS, EAA - 유명한 detection과 recognition을 연결하여, end-to-end방식으로 train한다.
Mask TextSpotter - sematic segmentation 문제로써 recognition을 처리한다.
```

*Character-level text detectors*
```
대부분 방식은 word를 기본 단위로 감지하는데, 의미나 공간 또는 색깔로 나눠진 word는 범위를 정의하는것이 쉽지 않다.
추가적으로 word의 boundary를 정확하게 정의할 수 없다.
이러한 점 때문에 regression과 segmentation방식에서 ground truth의 의미를 약화시킨다.

Seglink - character-level prediction대신 text grids를 찾고 추가적인 link prediction으로 이를 연결한다.
Mask TextSpotter - character-level확률 맵을 예측하지만, recognition에서만 사용된다.
WordSup - weakly supervised framework를 이용해 character-level 감지기를 학습한다. 
하지만, character representation이 rectangular이므로 perspective deformation에 취약하다.
또한, backbone구조에 제한이 있다.(SSD를 사용하며 많은 anchor boxes와 크기에 제한된다.)
```

3. Methodology
```
이 모델의 주요 목적은 자연 이미지에서 각각의 characters를 정확히 localize하는 것이다.
이를 위해, character region과 characters사이의 affinity를 예측하기 위해 deep neural network를 학습한다.
이용가능한 character-level데이터 셋이 부족하므로, weakly-supervised 방식으로 학습한다.
```

* Architecture

<img width="400" alt="스크린샷 2021-06-06 오후 5 14 54" src="https://user-images.githubusercontent.com/80749934/120917546-d1d55c80-c6ea-11eb-8ff5-186a9b42d76b.png">

```
backbone network는 VGG-16기반 fully convolutional network architecture이다. (batch normalization 포함)
decoding부분은 U-net과 유사한 skip connections을 가진다.
최종 output은 region score와 affinity score인 2개의 채널을 가진다.
```

* Training

*Ground Truth Label Generation*

<img width="600" alt="스크린샷 2021-06-06 오후 5 10 28" src="https://user-images.githubusercontent.com/80749934/120917437-4f4c9d00-c6ea-11eb-9c7d-dd0e74773b49.png">


```
character-level bounding boxes로 region score, affinity score에 대한 ground truth를 생성한다.

pose estimation
binary segmentation map과 다르게, Gaussian heatmap을 이용해 character의 중심 확률을 인코딩한다.
이 heatmap representation은 제한이 없는 ground truth region을 다룰 때 유용하다.
이러한 heatmap representation은 pose estimation 작업과 같은 곳에서 사용되었다.

Gaussian map
위 그림은 synthetic image에 대한 label 생성 파이프라인을 요약한다.
bounding box안에 각 픽셀마다 Gaussian distribution값을 직접 계산하는 것은 많은 시간이 걸린다.
character bounding boxes는 perspective projections에 의해 왜곡된다.
따라서, 다음 단계를 통해 region score와 affinity score에 대한 ground truth를 생성한다.
1. 2차원 isotropic Gaussian map을 준비한다.
2. Gaussian map과 character box간의 perspective transform을 계산한다.
3. Gaussian map을 box공간으로 변환한다(warp).

affinity boxes
위 그림 처럼 affinity boxes는 인접한 두 개의 character boxes를 이용해 정의된다.
각 chracter box의 모서리를 대각선으로 연결하면 위, 아래 2개의 삼각형이 만들어진다.
그런 다음, 한 쌍의 character box의 위, 아래 삼각형의 중심을 box의 모서리로써 연결하여 affinity box를 생성한다.

제안된 ground truth정의는 small receptive fields임에도 모델이 크고 긴 text를 감지 할 수 있게 한다.
character-level감지는 convolutional filters가 전체의 text instance대신,
오직 문자와 문자 간격에 집중할 수 있게 한다.
```

*Weakly-Supervised Learning*

```
synthetic 데이터 셋과 달리, 대부분 dataset은 word-level annotation을 가진다.
따라서, weakly-supervised 방식으로 word-level annotation으로 부터 character boxes를 생성한다.
이렇게 얻은 character boxes는 불완전하기 때문에 신뢰도를 계산하여 적용한다.
word box에 대한 신뢰도는 detected characters의 수와 ground truth characters의 수로 계산한다.
```

<img width="600" alt="스크린샷 2021-06-06 오후 6 28 58" src="https://user-images.githubusercontent.com/80749934/120919550-2f6ea680-c6f5-11eb-9f2e-c905c0cb582f.png">

```
위 그림은 characters를 분할하는 전체 순서를 보여준다.
1. original image에서 word-level images를 자른다.
2. 최신 학습 모델로 region score를 예측한다.
3. watershed algorithm을 이용해 character regions을 나눈다,
4. character boxes의 좌표를 cropping단계로 부터 얻은 inverse transform를 이용해 original image 좌표로 변환한다.
region score와 affinity score에 대한 pseudo-ground truths는,
획득한 quadrilateral character-level bounding boxes를 이용해 생성된다. 

모델이 weak-supervision을 이용해 학습할 때, 우리는 불완전한 pseudo-GTs로 학습할 수 밖에 없다.
만약 부정확한 region scores로 학습하면, 출력이 character regions이 blur될 수 있다.
이를 예방하기 위해, 모델에 의해 생성된 pseudo-GTs의 quality를 측정한다.
다행히, text annotation안에 word length라는 중요한 단서가 있다.
대부분 데이터 셋에 words의 필사가 제공되며, word의 길이가 pseudo-GTs의 신뢰도를 구하는데 사용된다.
```

<img width="400" alt="스크린샷 2021-06-06 오후 7 01 25" src="https://user-images.githubusercontent.com/80749934/120920475-f258e300-c6f9-11eb-8ece-7e1bb1e5b95d.png">
<img width="330" alt="스크린샷 2021-06-06 오후 7 01 37" src="https://user-images.githubusercontent.com/80749934/120920482-fc7ae180-c6f9-11eb-9653-7e31035bf003.png">
<img width="400" alt="스크린샷 2021-06-06 오후 7 01 48" src="https://user-images.githubusercontent.com/80749934/120920490-0ac8fd80-c6fa-11eb-8cb5-0c49e8fa7024.png">

|기호|definition|
|----|----|
|![](https://latex.codecogs.com/gif.latex?w)|a word-level training data|
|![](https://latex.codecogs.com/gif.latex?R%28w%29)|![](https://latex.codecogs.com/gif.latex?w)의 영역|
|![](https://latex.codecogs.com/gif.latex?l%28w%29)|![](https://latex.codecogs.com/gif.latex?w)의 길이|
|![](https://latex.codecogs.com/gif.latex?l%5Ec%28w%29)|추정된 ![](https://latex.codecogs.com/gif.latex?w)의 길이|
|![](https://latex.codecogs.com/gif.latex?s_%7Bconf%7D%28w%29)|식 1로 계산된 신뢰도|
|![](https://latex.codecogs.com/gif.latex?S_c%28p%29)|식 2로 계산된 픽셀별 신뢰도|
|![](https://latex.codecogs.com/gif.latex?S_r%5E*%28p%29)|the pseudo-GT region score|
|![](https://latex.codecogs.com/gif.latex?S_a%5E*%28p%29)|the pseudo-GT affinity score|
|![](https://latex.codecogs.com/gif.latex?S_r%28p%29)|the predicted region score|
|![](https://latex.codecogs.com/gif.latex?S_a%28p%29)|the predicted affinity score|
|![](https://latex.codecogs.com/gif.latex?L)|식 3, the objective function|

<img width="474" alt="스크린샷 2021-06-06 오후 8 25 46" src="https://user-images.githubusercontent.com/80749934/120922690-8c725880-c705-11eb-9b6a-26d2345be30f.png">

```
학습이 진행 되면서, model을 더욱 정확하게 예측하고 신뢰도가 서서히 증가하게된다.
초기에는 region score가 자연 상태에 친숙하지 않은 text에 대해 상대적으로 낮다.
모델은 불규칙한 fonts의 모습, SynthText데이터 셋에 대해 다른 데이터 분포를 가진 synthesized texts을 학습한다.

만약 신뢰도가 0.5미만일 경우, character의 가로가 일정하다 가정하고,
word region을 character의 수로 나누어 character-level predictions을 계산한다.
그리고 신뢰도는 0.5로 설정하여 보이지 않는 text의 모습을 학습한다.
```

* Inference

```
inference단계에서, 최종 출력은 word boxes, character boxes 및 polygons과 같은 다양한 모양들이 될 수 있다.
ICDAR과 같은 데이터 셋에 대한 evaluation protocol은 word-level IOU이다.

bounding boxes을 찾는 후처리는 다음과 같다.
1. 이진맵 M을 0으로 초기화한다.
2. 예측된 픽셀별 region score, affinity score가 각각 region threshold, affinity threshold보다 크면, 이진맵을 1로 설정한다.
3. 2에서 얻은 이진맵 M에 Connected Component Labeling(CCL)을 수행한다.
4. OpenCV에서 제공되는 connectedComponents와 minAreaRect를 적용한다.

CRAFT의 장점은 Non-Maximum Suppression(NMS)같은 후처리를 할 필요가 없다는 것이다.
CCL에 의해 분리된 word regions을 가져, word에 대한 bounding box는 쉽게 단일 enclosing rectangle로 정의된다.
pixel-level에서 수행되는 character linking처리가 text간의 관계를 검색하는데 의존하는 다른 linking방식과 다르다.
```

<img width="400" alt="스크린샷 2021-06-06 오후 9 18 46" src="https://user-images.githubusercontent.com/80749934/120924078-da3e8f00-c70c-11eb-8622-ef0a5c1590b1.png">

```
추가적으로, curved texts를 효과적으로 다루기 위해 전체 character영역 주위에 polygon을 생성 할 수 있다.
위 그림은 polygon 생성 과정이다.
1. scanning direction을 따라 character regions의 local maxima line을 찾는다.(파란색 화살표)
2. 불규칙한 결과를 예방하기 위해, local maxima lines의 길이를 이들 중 가장 긴 것으로 설정한다. 
3. local maxima lines의 중심점을 모두 이은 것을 center line이라 한다.(노란색)
4. characters의 기울기를 반영하기 위해 center line에 수직이 되게 local maxima lines을 회전시킨다.(빨간색 화살표)
5. local maxima lines의 끝 점은 text polygon의 control points에 대한 후보가 된다.
6. 바깥쪽 local maxima lines을 center line을 따라 이동시켜 마지막 control points를 만든다.
```



4. Experiment

* Datasets

|이름|이미지 수|언어|Annotation|
|----|----|----|----|
|ICDAR2013|train : 229, test : 233|English|word-level, rectangular boxes|
|ICDAR2015|train : 1000, test : 500|English|word-level, quadrilateral boxes|
|ICDAR2017|train : 7200, validation : 1800, test : 9000|9 languages|word-level, quadrilateral boxes|
|MSRA-TD500|train : 300, test : 200|English, Chinese|line-level, rotated rectangles|
|TotalText|train : 1255, test : 300|-|word-level, polygons|
|CTW-1500|train : 1000, test : 500|-|word-level, polygons(14 vertices)|

* Training strategy
```
공통
1. SynthText dataset으로 50k iteration만큼 훈련시킨다. 그리고 각각의 데이터셋을 fine-tuning시킨다.
2. ICDAR2015, 2017에 'DO NOT CARE'는 무시되며 신뢰도는 0이 된다.
3. ADAM optimizer를 사용한다.
4. multi gpu를 사용하는데, 하나는 training용 gpu이고 다른 하나는 supervision용 gpu이다. 
5. supervision gpu에 의해 생성된 pseudo-GT는 메모리에 저장된다.
6. fine-tuning하는 동안, Character영역을 확실히 구분하기 위해 SynthText를 1:5 비율로 사용한다. 
7. 문자와 비슷한 texture를 걸러내기 위해 On-line Hard Negative Mining을 1:3으로 적용한다.
8. crop, rotation, color variation같은 data augmentation적용 한다.

fine-tuning
1. Weakly-supervised training은 잘린 이미지에서 사각형과 단어의 길이 annotation이 필요하다.
2. 데이터 셋 IC13, IC15, IC17은 위 1조건을 만족하나, 나머지는 만족하지 않는다.
3. MSRA-TD500은 필사가 제공되지 않으며, TotalText와 CTW-1500은 polygon annotation만 제공된다.
4. 따라서, CRAFT를 ICDAR에만 train 시킨다. 나머지는 fine-tuning없이 테스트 된다.
4. 2개의 모델이 train되는데, IC15검증용 모델은 IC15로, 나머지 검증용은 IC13과 IC17로 train된다.
5. fine-tuning의 iteration수는 25k이다.
```

* Experimental Results

*Quadrilateral-type datasets(ICDARs, and MSRA-TD500)*
```
1. 모든 실험은 단일 이미지 해상도로 수행된다.
2. IC13, IC15, IC17 그리고 MSRA-TD500의 긴 변을 각각 960, 2240, 2560, 1600으로 resized한다.
3. end-to-end방식에서 공정함을 위해 original논문을 참고하여 detection-only결과 만을 가져온다.
4. CRAFT는 IC13에서 8.6FPS로 수행되며, 간단하지만 효과적인 후처리 덕분에 비교적 빠르다.
5. MSRA-TD500은 fine-tuning없이 수행되었으나, 결과적으로 다른 방식들을 능가한다.
```

*Polygon-type datasets(TotalText, CTW-1500)*
```
공통
1. 다양한 모양의 annotations은, weakly-supervised training시 character분할을 복잡하게 한다.
2. 위에 1번과 같은 이유로 IC13과 IC17로 train을 하며, TotalText와 CTW-1500은 실시 되지 않는다.
3. inference 단계에서, region score을 이용해 polygon-type annotation을 후 처리를 한다.
4. 단일 이미지 해상도로 수행되며, 각각 긴 변을 TotalText는 1280, CTW-1500은 1024로 resized한다.

CTW-1500 LinkRefiner
1. CTW-1500 dataset경우에 annotation이 line-level, 임의의 모양인 두가지 어려운 특성이 공존한다.
2. 이를 위해, small line refinement network(LinkRefiner)를 CRAFT와 함께 사용한다.
3. LinkRefiner의 input은 region score, affinity score 그리고 중간 feature map의 concatenation이다.
4. output은 긴 문장에 대한 refined affinity score이다.
5. characters를 결합하기 위해, refined affinity score는 기존 affinity score 대신 사용된다.
6. polygon생성은 TotalText에서 수행된 것과 같은 방식으로 수행된다.
7. LinkRefiner는 CRAFT를 freezing하고 CTW-1500에 대해 학습된다.

CRAFT의 character localization 기능은 임의의 모양의 글자를 감지하는데 다른 모델과 비교하여 강력하고 우수하다.
특히, TotalText는 quadrilateral-based text감지기로 inference가 불가능한 다양한 모양을 가졌는데 
매우 제한된 방법만이 이러한 데이터셋에서 검증된다.
```

* Discussions

```
- Robustness to Scale Variance
글자의 크기가 매우 다양함에도 불구하고, 모든 dataset에서 단일 크기로 실험을 수행했다.
이는 글자의 크기 변화 문제에 대해 multi-scale tests에 의존하는 대부분의 다른 방식과 다르다.
이러한 이점은 각각의 characters를 localizing하는 CRAFT 방식의 특성에서 비롯된다.
상대적으로 small receptive field는 큰 이미지안에 단일 character를 수용하는데 충분하다.
small receptive field의 그러한 특성은 CRAFT가 다양한 크기의 글자를 감지하는데 강력하게 만든다.

- Multi-language issue
IC17데이터 셋은 Bangla, Arabic characters를 포함하지만, synthetic text데이터 셋에 포함되어 있지않다.
게다가, 이러한 언어는 모든 character가 필기체로 쓰여지기 때문에 각각의 characters로 분류하는데 어렵다.
그러므로, CRAFT는 Latin, Korean, Chinese, and Japanese만큼 Bangla, Arabic characters를 잘 감지할 수 없다.
동아시아 Characters의 경우, 일정한 크기로 나누는게 쉽다.
그러한 특징은 weakly-supervision을 통한 모델을 train하는데 도움이 된다.

- Comparison with End-to-end method
text를 감지하는데만 ground truth boxes를 통해 train된다.
실패 사례 분석에서, CRAFT모델이 visual cues보다 recognition결과로 부터 이익을 얻는 것을 기대한다.

- Generalization ability
추가적인 fine-tuning없이 3가지 다른 데이터셋에서 최첨단 성능을 달성했다.
이것은 CRAFT가 일반적인 texts의 특징을 capture할 수 있는 것을 입증한다.
```

5. Conclusion
```
character-level annotation없이 각각의 character를 감지할 수 있다.
다양한 모양의 text를 덮는 character region score, character affinity score를 제안했다.
실제 데이터 셋들은 character-level annotation이 희박하므로,
interim model로 부터 pseudo-ground truthes를 생성하는 weakly-supervised learning 방식을 제안 했다.
CRAFT는 대부분의 데이터 셋에서 최첨단 성능을 보여주며, fine-tuning없이 이러한 성능을 보여줌으로써 일반화 능력을 입증한다.
더 좋은 성능을 위해 end-to-end방식으로 recognition모델과 함께 train할 것이다.
```

6. 공부할 것들

```
hand-crafted features
Regression-based text detectors
End-to-end text detectors
anchor boxes
sliding windows
U-net
pose estimation
watershed algorithm
Attention mechanism
Connected Component Labeling
connectedComponents, minAreaRect
Non-Maximum Suppression(NMS)
isotropic
LinkRefiner
OHEM
receptive field - 출력 이미지의 픽셀 하나(출력 뉴런)에 영향을 미치는 입력 이미지의 픽셀(입력 뉴런) 크기
visual cues
scene text spotting
```

