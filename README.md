# AI Study Python Project

AI 스터디를 위한 파이썬 프로젝트입니다. TensorFlow를 활용한 머신러닝 기초 예제를 포함하고 있습니다.

## 포함된 파일

- `mlexample1.py`: TensorFlow를 이용한 선형 회귀(Linear Regression) 예제 코드입니다.

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

## 실행 방법

### 요구 사항
- Python 3.x
- TensorFlow 2.x

### 설치
```bash
pip install tensorflow
```

### 실행
```bash
python mlexample1.py
```
