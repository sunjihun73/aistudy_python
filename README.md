# Python Project for AI Study 

AI 스터디를 위한 파이썬 프로젝트입니다. TensorFlow를 활용한 머신러닝 기초 예제를 포함하고 있습니다.

## 포함된 파일

- `mlexample1.py`: TensorFlow를 이용한 선형 회귀(Linear Regression) 예제 코드입니다.
- `mlexample2.py`: TensorFlow/Keras를 활용한 MNIST 손글씨 숫자 분류(Handwritten Digit Classification) 예제 코드입니다.

## mlexample1.py 설명

이 코드는 단순 선형 회귀 모델을 학습시키는 예제입니다.

- **데이터**: `X = [1, 2, 3]`, `Y = [1, 2, 3]` (X와 Y가 동일한 정비례 관계)
- **가설(Hypothesis)**: `H(x) = Wx + b`
- **비용 함수(Cost Function)**: 평균 제곱 오차(Mean Squared Error)
- **옵티마이저**: 경사 하강법(Stochastic Gradient Descent, SGD)
- **학습 횟수**: 2000회 이상

### 주요 기능
- 100회 학습마다 가중치(Weight), 편향(Bias), 비용(Cost)을 출력합니다.
- 학습 완료 후 입력값에 따른 예측 결과를 출력합니다.

## mlexample2.py 설명

이 코드는 MNIST 데이터셋을 사용하여 손글씨 숫자를 분류하는 다중 클래스 분류 예제입니다.

- **데이터**: MNIST (28x28 픽셀의 손글씨 숫자 이미지 70,000장)
- **모델 구조**:
  - `Flatten`: 28x28 입력을 784차원 벡터로 변환
  - `Dense`: 10개의 출력 노드와 Softmax 활성화 함수를 사용하여 0~9까지의 확률 계산
- **옵티마이저**: Adam
- **손실 함수(Loss Function)**: Sparse Categorical Crossentropy
- **학습 파라미터**: Epochs 15, Batch Size 32

### 주요 기능
- 학습 완료 후 테스트 데이터셋을 통해 모델의 정확도를 평가합니다.
- 무작위로 선택된 테스트 이미지 1장을 화면에 출력하고, AI가 예측한 결과를 보여줍니다.

## 실행 방법

### 요구 사항
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

### 설치
```bash
pip install tensorflow numpy matplotlib
```

### 실행
```bash
# 선형 회귀 예제 실행
python mlexample1.py

# MNIST 분류 예제 실행
python mlexample2.py
```
