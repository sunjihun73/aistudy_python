
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

def mlFunc(name):
  # Load MNIST dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Preprocess the data
  x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
  x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

  # Convert labels to one-hot encoding
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

  # Build the CNN model
  model = tf.keras.Sequential(
    [
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(625, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ]
  )

  # Compile the model
  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

  # Train the model
  model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

  # 테스트 데이터셋으로 모델 성능 평가하기
  test_loss, test_acc = model.evaluate(x_test, y_test)
  print(f"\n정확도 평가결과: {format(test_acc * 100, '4.2f')} % \n")

  # 예제 그림 띄워보기
  r = random.randint(0, 9999)
  plt.imshow(x_test[r], cmap='gray')
  # plt.show()
  plt.savefig('result.png')
  print("\n[알림] 예제 그림이 'result.png' 파일로 저장되었습니다.\n")

  # 예제 그림 맞추기
  example_image = x_test[r]
  example_image = np.expand_dims(example_image, axis=0)  # Add an extra dimension for batch
  predictions = model.predict(example_image)
  predicted_class = np.argmax(predictions[0])
  print(f"\n\n 이 그림은 무슨 글자이지? \n ==> AI가 예측한 답 : {predicted_class}")



# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    mlFunc('sunjihun')