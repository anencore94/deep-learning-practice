"""
tf 기본 API
"""
import tensorflow as tf

print(tf.__version__)

# Tensor
# 상수형 텐서 생성
print(tf.ones(shape=(2, 2)))

print(tf.random.normal(shape=(2, 2)))

# Variables
# tf.Variables 는 각 원소를 변경할 수 있는 텐서 생성
# 단, 초기값은 특정 const tensor 로 지정해주어야 함

a = tf.Variable(tf.random.normal(shape=(2, 2)))
print(a)

a.assign(a * 10)
print(a)

# Tensorflow 에서 수학을 하는 것
# numpy 의 api 와 동일하지만, tf 의 코드는 GPU 와 TPU 상에서 실행 가능
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))
print("a is\n", a)
print("b is\n", b)

c = a + b
d = tf.square(c)
e = tf.exp(d)
print("c is\n", c)
print("d is\n", d)
print("e is\n", e)

# GradientTape 를 사용한 미분
# API 가 조금 특이
# with tf.GradientTape() 로 tape 객체를 하나 만든 다음,
# 미분할 target 함수를 tape.watch(source) 이후에 정의해야 함
# 그다음 tape.gradient() 함수가 호출되면 tape 객체 리소스 자원은 해제됨
# 물론 gradient 를 여러 변수에 대해 계산하고 싶을 수 있으므로 persistent gradient tape
# tf.GradientTape(persistent=True) 로 선언 가능
with tf.GradientTape() as tape:
  tape.watch(a)  # tensor a 에 적용되는 연산의 히스토리에 대한 기록을 시작
  c = tf.sqrt(tf.square(a) + tf.square(b))
  dc_da = tape.gradient(c, a)  # dc/da, c 를 a 에 대해 gradient 구한 것

print("dc_da is", dc_da)

# a 가 tf Variable 이면 자동으로 watch 가 적용되어 있어 tape.watch(a) 필요 x
a = tf.Variable(a)
with tf.GradientTape() as t:
  c = tf.sqrt(tf.square(a) + tf.square(b))
  dc_da_2 = t.gradient(c, a)

print("dc_da is", dc_da_2)

# 고차 미분 가능
with tf.GradientTape() as outer_tape:
  with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
  # gradient 의 target parameter 로 tape.gradient 의 return 객체를 쓸 수 있음
  d2c_da2 = outer_tape.gradient(dc_da, a)

print("d2c_da2 is", d2c_da2)
