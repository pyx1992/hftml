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


def main(argv):
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


if __name__ == '__main__':
  app.run(main)
