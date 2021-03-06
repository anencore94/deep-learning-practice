import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
  """
  Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

  Input 으로 (vector_1, vector_2) 받아서,
  N(vector_1, vector_2*I) 의 sample 을 return
  """

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """
  input 받아서 hidden layer 하나 태우고
  해당 hidden layer 의 output 을 각각 dense_mean layer 와 dense_log_var layer 에
  태워서 얻은 두 개의 ouput : z_mean, z_log_var 와
  이 두 z_* 으로부터 sampling 해서 얻은 z 를 반환하는 Layer
  """

  def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder",
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self, original_dim, intermediate_dim=64, name="decoder",
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
    self.dense_output = layers.Dense(original_dim, activation="sigmoid")

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
  """
  Combines the encoder and decoder into an end-to-end model for training.

  Input 하나 받고, encoder, decoder 거치면서 loss 저장하고, output 반환
  """

  def __init__(
          self,
          original_dim,
          intermediate_dim=64,
          latent_dim=32,
          name="autoencoder",
          **kwargs
  ):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(
      z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    self.add_loss(kl_loss)  # loss 저장
    return reconstructed
