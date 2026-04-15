
# 필요한 라이브러리 가져오기
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

def mlFunc(name):
    # MNIST 데이터셋을 로드하기
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # 데이터 사전 처리하기 - Normalize the pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 모델 아키텍처 정의
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),  # 784개의 변수로 나열한다.
            keras.layers.Dense(10, activation='softmax')  # 10개의 클래스를 가진 softmax 함수로 연결한다.
        ]
    )

    # 모델 컴파일 및 훈련
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=15, batch_size=32)

    # 테스트 데이터셋으로 모델 성능 평가하기
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\n정확도 평가결과: {format(test_acc * 100, '4.2f')} % \n")

    # 예제 그림 띄워보기
    r = random.randint(0, 9999)
    plt.imshow(test_images[r], cmap='gray')
    plt.show()

    # 예제 그림 맞추기
    example_image = test_images[r]
    example_image = np.expand_dims(example_image, axis=0)  # Add an extra dimension for batch
    predictions = model.predict(example_image)
    predicted_class = np.argmax(predictions[0])
    print(f"\n\n 이 그림은 무슨 글자야? \n ==> AI가 예측한 답 : {predicted_class}")


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    mlFunc('sunjihun')