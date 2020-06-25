"""
np, tf 기본
"""

import numpy as np
import tensorflow as tf

# 기본 정보
print("=" * 15)
print("STEP 1")
print("=" * 15)

# 현재 tf package 의 버전
print(tf.__version__)

# eager mode 가 현재 켜져있는지
print(tf.executing_eagerly())

# 텐서 선언
print("=" * 15)
print("STEP 2")
print("=" * 15)

# tensor 선언 방법
m2 = np.array([[1.0, 2.0],
               [3.0, 4.0]], dtype=np.float32)
m3 = tf.constant([[1.0, 2.0],
                  [3.0, 4.0]])

print(type(m2))  # m2 는 tensor 가 아님
print(type(m3))

# m2 를 tensor 로 변환 (t2)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)

print(type(t2))
print(type(m3))

print(t2)
# 원래 tf 1.x 대에서는 tensor 객체는 session.run 이후에 print 가능했지만
# tf 2.x 대에서는 eager mode 가 기본이므로 바로 print 가능
print(m3)

# np 와 tf 호환
# Numpy 와 tensorflow 간의 연산은 서로 호환 가능
print("=" * 15)
print("STEP 3")
print("=" * 15)

ndarray = np.ones([3, 3])
print(ndarray)

print("\nTf operation convert np array to tensor automatically")
tensor = tf.multiply(ndarray, 3)
print(tensor)

print("\nNp operation convert tensor to np array automatically")
print(np.add(tensor, 1))

print("\n.numpy() method convert tensor to np array")

print(tensor.numpy())

print("\nconstant method create tensor")
a = tf.constant(1.5)
b = tf.constant([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.numpy())
print(b)
print(b.numpy())
print(type(b.numpy()))

# tf.Variables
weight = tf.Variable(tf.random_normal_initializer(stddev=0.1)(shape=[2, 2]))
print(weight)

# tf.data.Dataset
# 여러 tf 연산을 쉽게 하기 위해 보통 np array 을 tf.data.Dataset 으로 변환해서 사용
# normalize 등의 데이터 전처리 하고싶은 경우 사용

a = np.arange(10)
print(a)

ds_tensors = tf.data.Dataset.from_tensor_slices(a)
ds_tensors = ds_tensors.map(tf.square).shuffle(20).batch(2)
# batch(n) : 원소를 n개씩 묶은 걸 하나의 원소로 하는 리스트로 만드는 연산

# 보통 데이터 처리할 때 epoch, batch 로 나눠서 이중 for 문 돌림
print("whole data transformation started")
for _ in range(3):  # epoch : 전체 data 가 다 들어가면 1 epoch
  for x in ds_tensors:  # batch : ds_tensors 에 대해서 2개 씩 끊은 각각의 원소가 x
    print(x)
  print("one shuffle finished")
print("whole data transformation finished")

print("한 번 더 transformation 수행")
for element in ds_tensors.as_numpy_iterator():
  print(element)
