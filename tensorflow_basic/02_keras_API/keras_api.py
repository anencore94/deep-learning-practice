"""
tf 기본 API
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential

print(tf.__version__)


# 1) Layer 기본 클래스
# Layer 클래스를 상속하여 Linear Layer 를 만듦
class Linear(Layer):
  """y = w.x + b"""

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  # Layer 의 build 메소드 오버라이딩
  # 첫 번째 입력의 shape 이 확인되는 순간 호출되는 lazy 한 메소드
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)  # GradientTape 가 watch 가능
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


# 우리가 만든 Layer객체를 인스턴스화 합니다
linear_layer = Linear(units=4)
y = linear_layer(tf.ones((3, 2)))  # 여기서 build 가 호출
# input : tf.ones((3, 2) : 3 x 2
#  1   1   1
#  1   1   1
# x_1 x_2 x_3
# 이런 식으로 x_1(1,1), x_2(1,1), x_3(1,1) 3 개의 2 차원 input

# x * w 의 차원 : (3 x 2) * (2 x 4) = (3 * 4)
print(y.shape)


# 2) 학습 가능한, 그리고 학습 불가능한 가중치
class ComputeSum(Layer):
  """입력의 합산 결과를 반환하는 Layer"""

  def __init__(self, input_dim):
    super(ComputeSum, self).__init__()
    # 학습 불가능한 가중치를 생성합니다.
    self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                             trainable=False)  # 학습 불가능

  # ComputeSum 의 인스턴스가 불릴 때마다 실행되는 메소드
  def call(self, inputs):
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return self.total


my_sum = ComputeSum(2)
x = tf.ones((4, 2))

print(my_sum(x).numpy())  # [4. 4.]
my_sum(x)  # [8., 8.]
print(my_sum(x).numpy())  # [12. 12.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []


# 3) 재귀적으로 Layer 를 조합하는 것
class MLP(Layer):
  """Linear Layer 의 간단한 층을 쌓는 Layer 입니다."""

  def __init__(self):
    super(MLP, self).__init__()
    self.linear_1 = Linear(32)
    self.linear_2 = Linear(32)
    self.linear_3 = Linear(10)

  def call(self, inputs):
    first_out = self.linear_1(inputs)
    first_out = tf.nn.relu(first_out)
    second_out = self.linear_2(first_out)
    second_out = tf.nn.relu(second_out)
    return self.linear_3(second_out)


mlp = MLP()

# `mlp` 객체 처음 호출하면 Linear(Layer) 의 build 가 불려서 그 때, 가중치 생성
y = mlp(tf.ones(shape=(3, 64)))

# 가중치들은 재귀적으로 추적됩니다.
# linear_1 의 가중치 2개 (w_1, b_1), linear_2 의 가중치 2개 linear_3 의 가중치 2개
assert len(mlp.weights) == 6


# 4) call method 의 training 인자
class Dropout(Layer):
  def __init__(self, rate):
    super(Dropout, self).__init__()
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
      return tf.nn.dropout(inputs, rate=self.rate)
    return inputs


class MLPWithDropout(Layer):
  """Linear Layer, DropoutLayer 를 혼합한 간단한 층을 쌓는 Layer 입니다."""

  # init 은 변수 선언만 해주고
  def __init__(self):
    super(MLPWithDropout, self).__init__()
    self.linear_1 = Linear(32)
    self.dropout = Dropout(0.5)  # dropout 의 rate
    self.linear_3 = Linear(10)

  # call 이 각 layer 를 연결하는 실제 로직을 작성하는 메소드
  def call(self, inputs, training=None):
    first_out = self.linear_1(inputs)
    first_out = tf.nn.relu(first_out)
    second_out = self.dropout(first_out, training=training)
    return self.linear_3(second_out)


mlp = MLPWithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)  # 여기서 MLPWithDropout.call 이
# 불리고 dropout 의 training = True 로 선언해줌
y_test = mlp(tf.ones((2, 2)), training=False)

# Sequential class 는 Layer 의 목록을 단순하게 이어붙인 Model 로 변환해줌
model = Sequential([Linear(32), Dropout(0.5), Linear(10)])

y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)

# 5) Loss 클래스
bce = tf.keras.losses.BinaryCrossentropy()
y_true = [1., 0., 1., 1.]  # 목표 (레이블)
y_pred = [1., 1., 1., 0.]  # 예측 결과

loss = bce(y_true, y_pred)  # BinaryCrossEntropy 가 상속하는 LossFunctionWrapper
# 의 call 함수에 따라서 작동
print(loss)  # loss(bce 의 반환형)은 scalar tensor
print('손실:', loss.numpy())

# 6) Metric 클래스
# Encapsulates metric logic and state.
m = tf.keras.metrics.AUC()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])  # true, predict 업데이트 가능
print('중간 결과: ', m.result().numpy())

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print('최종 결과: ', m.result().numpy())
