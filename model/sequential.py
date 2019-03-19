# 2019
# author: yuxuan

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


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

  def save_model(self, save_path):
    self._model.save(save_path)

  def load_model(self, load_path):
    self._model = keras.models.load_model(load_path)
    print(self._model.summary())


class SequentialRegressor(object):
  def __init__(self):
    self._model = None

  def _build_model(self, input_shape):
    model = keras.Sequential([
      layers.Dense(256, activation=tf.nn.leaky_relu, input_shape=[input_shape]),
      layers.Dense(512, activation=tf.nn.leaky_relu),
      layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.00015)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    self._model = model

  def train_model(self, x, y):
    self._build_model(x.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    self._model.fit(X_train, y_train, epochs=150)
    loss, mae, mse = self._model.evaluate(X_test, y_test)
    print('Test loss, mae, mse: ', loss, mae, mse)

  def predict(self, x):
    predictions = self._model.predict(x)
    return predictions

  def save_model(self, save_path):
    self._model.save(save_path)

  def load_model(self, load_path):
    self._model = keras.models.load_model(load_path)
    print(self._model.summary())
