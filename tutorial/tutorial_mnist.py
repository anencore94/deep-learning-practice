"""
This script is the tutorial for using tensorflow API

@ Link : https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

# STEP 1 | data 준비
# MNIST data 준비
mnist = tf.keras.datasets.mnist
# local 의 ~/.keras/datasets/mnist.npz 파일 생성
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# STEP 2 | data 확인
# 보통 shape 을 기본적으로 확인
print("shape: {}, dimension: {}, dtype:{}, len:{}".
      format(x_train.shape, x_train.ndim, x_train.dtype, len(x_train)))
print("Array's Data:\n", x_train)

print("shape: {}, dimension: {}, dtype:{}, len:{}".
      format(y_train.shape, y_train.ndim, y_train.dtype, len(y_train)))
print("Array's Data:\n", y_train)

print("shape: {}, dimension: {}, dtype:{}, len:{}".
      format(x_test.shape, x_test.ndim, x_test.dtype, len(x_test)))
print("Array's Data:\n", x_test)

print("shape: {}, dimension: {}, dtype:{}, len:{}".
      format(y_test.shape, y_test.ndim, y_test.dtype, len(y_test)))
print("Array's Data:\n", y_test)

# unique element 목록 및 개수 확인
print("{}'s unique elements :{},\n count : {}".
      format("x_train", np.unique(x_train), len(np.unique(x_train))))
print("{}'s unique elements :{},\n count : {}".
      format("y_train", np.unique(y_train), len(np.unique(y_train))))

# STEP 3 | data preprocessing
# x_train, x_test normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# STEP 4 | modeling
# Sequential 은 NN 을 만들 건데, layer 를 param 로 입력한 순서대로 붙여서 만듦
model = tf.keras.models.Sequential([
  # Flatten : input data 를 하나의 column 으로 쫙 펴는 layer
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # Dense : 모든 unit (neuron) 이 연결된 layer
  tf.keras.layers.Dense(128, activation='relu'),
  # Dropout : overfitting 방지를 위해 input 의 20 % 는 버리는 layer
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# model 학습 시 optimizer, loss 등을 어떻게 설정할 것인지 정의
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model 확인
model.summary()

# STEP 5 | learning
# model 학습
model.fit(x_train, y_train, epochs=5)

# 완료 ack
print("model training finished")

# STEP 6 | evaluating
# test dataset 을 학습된 model 에 넣고,
# 이미 알고 있는 y_test 와 비교하여 모델의 정확도 확인
model.evaluate(x_test, y_test, verbose=2)
