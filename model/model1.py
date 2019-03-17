from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from absl import app, flags


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'training_data_path',
    '',
    '')
flags.DEFINE_string(
    'test_data_path',
    '',
    '')


def build_model(input_shape):
  model = keras.Sequential([
    layers.Dense(32, activation=tf.nn.relu, input_shape=[input_shape]),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


def test():
  x = pd.read_csv(FLAGS.training_data_path)
  nan_mask = np.any(pd.isnull(x), 1)
  x = x[~nan_mask]
  y_label = str(max([int(c) for c in x.columns]))
  y = x.pop(y_label)
  print(y.describe())
  model = build_model(x.shape[1])
  model.summary()
  epochs = 500
  history = model.fit(
    x, y,
    epochs=epochs, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])
  print('')
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  print(hist.tail())

  #loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
  #print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


def main(argv):
  import tensorflow as tf
  from tensorflow import keras
  from sklearn.model_selection import train_test_split
  x = pd.read_csv('up_down.csv')
  y_col = x.columns[-1]
  y = x.pop(y_col)

  model = keras.Sequential([
      keras.layers.Dense(16, activation=tf.nn.relu, input_shape=[len(x.columns)]),
      keras.layers.Dense(16, activation=tf.nn.relu),
      keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
  model.fit(X_train, y_train, epochs=15)

  test_loss, test_acc = model.evaluate(X_test, y_test)
  print('Test accuracy:', test_acc)


if __name__ == '__main__':
  app.run(main)
