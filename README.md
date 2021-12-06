# AlexNet

[](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

# 코드 구현

[](https://github.com/Soohyeon-Bae/AlexNet/blob/main/AlexNet_keras.py)

- 참고
    
    [CNN의 parameter 개수와 tensor 사이즈 계산하기](https://seongkyun.github.io/study/2019/01/25/num_of_parameters/)
    

# 논문 리뷰

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e6f63774-485b-4e1c-9a5f-eb585fd07ce7/Untitled.png)

- 총 8개의 레이어로 구성되며, 5개의 Convolutional Layer와 3개의 Fully-Connected Layer로 구성
- 마지막 Fully-Connected Layer의 활성화 함수는 softmax이며, 총 클래스 레이블은 1000개

# The Architecture

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/95c5e7aa-4fcb-4d03-9303-9c1925a68d95/Untitled.png)

## ReLU Nonlinearity

- 학습 속도 향상 → **미분 계산이 간단하여**(0이하는 0, 0이상은 1) 학습속도가 빨라짐
- 모든 레이어에 적용
- Non-saturating nonlinearity
    - **Saturating nonlinearities 함수**: 어떤 입력 x가 무한대로 갈 때 함수의 결과값이 **어떤 범위내에서만** 움직이는 함수 ex) sigmoid, tanh
    - **Non-saturating nonlinearity 함수**: 어떤 입력 x가 무한대로 갈 때 함수의 결과값이 **무한대로** 가는 함수 ex) ReLU

## Training on Multiple GPUs

- 커널을 병렬적으로 학습(GPU parallelization)
- GPU1에서는 색상과 관련 없는 정보를 학습하고 GPU2는 색상과 관련된 정보를 학습함
- 다른 layer들은 전 단계의 같은 채널의 특성맵들과만 연결되어 있는 반면 3번째 convolutional layer는 2번째 layer의 두 채널의 특성맵들과 모두 연결됨

## Local Response Normalization

(k = 2, n = 5, alpha = 10^-4, beta = 0.75)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36db1c64-33c7-49b1-b272-6514a694ed56/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3d707c6-4dff-4357-b4a8-43ddc7b4318a/Untitled.png)

- 첫번째, 두번째 convolutional layer에 적용
- Lateral inhibition
    
    신경세포들이 흥분하게 되면 옆에 있는 이웃 신경세포에 억제성 신경전달물질을 전달하여, 이웃 신경 세포가 덜 활성화되도록 함
    
    - ReLU는 max(a,0)과 같은 형태로, 양수 방향의 값을 그대로 사용하기 때문에 특정 pixel 값이 엄청나게 크다면 주변의 pixel도 영향을 받음
- 현재는 LPN 대신 **batch normalization**을 주로 사용함

## Overlapping Pooling

(filter size = 3 x 3, strides = 2)

- pooling layer에서 **stride보다 더 큰 filter**를 사용하여 겹쳐지도록 함
- 오버피팅 개선 → With overlapping regions, there is **less loss of surrounding spatial information**
    
    [Why do training models with overlapping pooling make it harder to overfit CNNs? (in Krizhevsky 2012)](https://www.quora.com/Why-do-training-models-with-overlapping-pooling-make-it-harder-to-overfit-CNNs-in-Krizhevsky-2012/answer/Amrit-Krishnan?ch=10&share=51fc9137&srid=SZvq)
    
- 첫번째, 두번째, 다섯번째 convolutional layer에 적용

# Reducing Overfitting

## Data Augmentation

- 이미지 translation 및 horizontal reflection 적용하여 224x224 사이즈로 random 추출
- 학습 이미지 데이터의 RGB channel 값을 변형

## Dropout (0.5 probability)

- 일부 neuron을 강제로 0으로 변경하여 각 neuron이 서로 다른 neuron에게 의지할 수 없게 하며, neuron간의 복잡한 상호 의존을 감소함

## Summary

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e055249-6d8f-46b2-8712-3a85932ca3c7/Untitled.png)

- 참고
    
    [[논문 요약4] ImageNet Classification with Deep Convolutional Neural Networks](https://arclab.tistory.com/156)
    
    [[Deep Learning] 딥러닝에서 사용되는 다양한 Convolution 기법들](https://eehoeskrap.tistory.com/431)
    
    [[CNN 알고리즘들] AlexNet의 구조](https://bskyvision.com/421)
    
    [AlexNet 구조 개인정리](https://wiserloner.tistory.com/1126)
    
    [[논문 구현] PyToch로 AlexNet(2012) 구현하기](https://deep-learning-study.tistory.com/518)
