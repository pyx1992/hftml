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
from sklearn.model_selection import train_test_split


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'training_data_path',
    '',
    '')
flags.DEFINE_string(
    'test_data_path',
    '',
    '')


class SequentialClassifier(object):
  def __init__(self):
    self._model = None

  def _build_model(self, input_shape):
    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[input_shape]),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    self._model = model

  def train_model(self, x, y):
    self._build_model(x.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    self._model.fit(X_train, y_train, epochs=10)
    test_loss, test_acc = self._model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

  def predict(self, x):
    predictions = self._model.predict(np.array(x))
    return np.argmax(predictions)


class SequentialRegressor(object):
  def __init__(self):
    self._model = None

  def _build_model(self, input_shape):
    model = keras.Sequential([
      layers.Dense(32, activation=tf.nn.relu, input_shape=[input_shape]),
      layers.Dense(32, activation=tf.nn.relu),
      layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    self._model = model

  def train_model(self, x, y):
    self._build_model(x.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    self._model.fit(X_train, y_train, epochs=10)
    test_loss, test_acc = self._model.evaluate(X_test, y_test)
    print('Test accuracy: ', test_acc)

  def predict(self, x):
    predictions = self._model.predict(np.array(x))
    return predictions[0]
