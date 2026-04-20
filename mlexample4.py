import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def mlFunc(name):
  text = """경마장에 있는 말이 뛰고 있다\n
  그의 말이 법이다\n
  가는 말이 고와야 오는 말이 곱다\n"""

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts([text])
  vocab_size = len(tokenizer.word_index) + 1

  sequences = list()
  for line in text.split('\n'):  # 줄바꿈 문자를 기준으로 문장 토큰화
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
      sequence = encoded[:i + 1]
      sequences.append(sequence)

  max_len = max(len(l) for l in sequences)  # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력

  sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

  sequences = np.array(sequences)
  X = sequences[:, :-1]
  y = sequences[:, -1]

  y = to_categorical(y, num_classes=vocab_size)

  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

  embedding_dim = 10
  hidden_units = 32

  model = Sequential()
  model.add(Embedding(vocab_size, embedding_dim))
  model.add(SimpleRNN(hidden_units))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X, y, epochs=200, verbose=0)

  def sentence_generation(model, tokenizer, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
      # 현재 단어에 대한 정수 인코딩과 패딩
      encoded = tokenizer.texts_to_sequences([current_word])[0]
      encoded = pad_sequences([encoded], maxlen=5, padding='pre')
      # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
      result = model.predict(encoded, verbose=0)
      result = np.argmax(result, axis=1)

      for word, index in tokenizer.word_index.items():
        # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면 break
        if index == result:
          break

      # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
      current_word = current_word + ' ' + word

      # 예측 단어를 문장에 저장
      sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence

  print("\n\n이 예제는 '딥 러닝을 이용한 자연어 처리 입문(유원준/안상준, https://wikidocs.net/45101)'의 \n예제이며, 일부 수정하였습니다.\n")
  print("\n입력된 단어를 사용하여 문장을 완성하는 모델입니다.")
  print("이 모델은 아래의 세 문장을 학습하였습니다.\n")
  print("\n<학습된 문장>\n경마장에 있는 말이 뛰고 있다\n그의 말이 법이다\n가는 말이 고와야 오는 말이 곱다\n\n")

  while True:
    text = input("'경마장에', '그의', '가는' 중 하나를 입력해보세요 (또는 'exit'를 입력하여 종료하세요): ")

    # Check if the user wants to exit
    if text == 'exit':
      break
    elif text == '경마장에':
      str_len = 4
    elif text == '그의':
      str_len = 2
    elif text == '가는':
      str_len = 5
    else:
      str_len = 1

    print(sentence_generation(model, tokenizer, text, str_len))


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    mlFunc('sunjihun')