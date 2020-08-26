import numpy as np
import tensorflow as tf

"""
x_1, x_2, x_3, x_4 가 input 으로 들어오면,

SimpleRNN layer + Dense layer 모델로
x_5 를 예측하는 모델
"""

# setup data
train_data = []  # sequential input : (x_1, x_2, x_3, x_4) 의 세트가 여러 개 있는 데이터
train_label = []  # label : x_5 가 여러 개 있는 데이터
for i in range(6):
  lst = list(range(i, i + 4))
  train_data.append(list(map(lambda c: [c / 10], lst)))
  train_label.append((i + 4) / 10)

train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array([[[0.8], [0.9], [1.0], [1.1]]])

print(f'train input data : {train_data}')
print(f'train output data : {train_label}')

# build model
n_neurons = 10  # 각 h_t : output(hidden) 의 각 timestamp vector 의 차원을 의미
model = tf.keras.Sequential([
  tf.keras.layers.SimpleRNN(
    units=n_neurons,  # RNN 에 존재하는 neuron 의 개수 : param 의 개수에 비례
    return_sequences=False,
    # 해당 layer 의 output 으로 h_(t+1) 만 출력할 것인지,
    # 아니면 h_1, h_2, ... h_t, h_(t+1) 전부 출력할 것인지
    input_shape=[4, 1]),  # train_set 의 하나의 row 의 timestamp 축을 제외한 차원
  tf.keras.layers.Dense(1)]
  # RNN 의 output 인 (None, n_neurons) 차원 벡터를 scalar 로 변환하는 층
)

# compile model
model.compile(optimizer='adam', loss='mse')

# see model info
model.summary()

# train model for epochs
model.fit(train_data, train_label, epochs=1000, verbose=0)

# test trained_model with train_data
print(model.predict(train_data))

# test trained_model with test_data different from train_data
print(model.predict(test_data))
