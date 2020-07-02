from vae_model import *

# data preprocess
original_dim = 784
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, original_dim).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# vae model setup
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()

# Iterate over epochs.
epochs = 2
for epoch in range(epochs):
  print("Start of epoch %d" % (epoch,))

  # Iterate over the 'batches' of the dataset.
  for step, x_batch_train in enumerate(
          train_dataset):  # batch, shuffle 이 이미 설정된 tf.data.Dataset 임
    with tf.GradientTape() as tape:
      reconstructed = vae(x_batch_train)
      # Compute reconstruction loss
      loss = mse_loss_fn(x_batch_train, reconstructed)
      loss += sum(vae.losses)  # Add KLD regularization loss

    # vae.trainable_weights 에 대한 loss ft 의 gradient 를 구하는 것
    grads = tape.gradient(target=loss, sources=vae.trainable_weights)
    # optimizer 로 gradient 구한다
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    loss_metric(loss)  # 현재까지 loss 의 평균

    if step % 100 == 0:
      print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
