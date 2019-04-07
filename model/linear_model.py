# 2019
# author: yuxuan

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.model_selection import train_test_split


class LinearModelSignals(object):
  def __init__(
      self, olsres, enter_buy_threshold=None, enter_sell_threshold=None,
      exit_buy_threshold=None, exit_sell_threshold=None):
    self._olsres = olsres
    self._enter_buy = enter_buy_threshold
    self._enter_sell = enter_sell_threshold
    self._exit_buy = exit_buy_threshold
    self._exit_sell = exit_sell_threshold
    assert self._enter_sell <= self._exit_buy
    assert self._exit_sell <= self._enter_buy
    self._pred = np.nan

  def on_feature(self, feature):
    x = np.array([feature])
    if np.any(np.isnan(x)):
      return
    #x = sm.add_constant(feature)
    self._pred = self._olsres.predict(x)

  def should_enter_buy(self):
    return self._pred > self._enter_buy

  def should_exit_buy(self):
    return self._pred < self._exit_buy

  def should_enter_sell(self):
    return self._pred < self._enter_sell

  def should_exit_sell(self):
    return self._pred > self._exit_sell


class LinearModel(object):
  def __init__(self):
    self._result = None

  def train_model(self, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    self._result = model.fit()
    print(self._result.summary())
    y_hat = self._result.predict(X_train)
    y_test_hat = self._result.predict(sm.add_constant(X_test))
    print('test y corr:', np.corrcoef(y_test, y_test_hat))
    y = pd.DataFrame({'y': y_train.tolist(), 'yhat': y_hat.tolist()})
    y_test = pd.DataFrame({'y': y_test.tolist(), 'yhat': y_test_hat.tolist()})
    y.to_csv('lm_y_train.csv', index=False)
    y_test.to_csv('lm_y_test.csv', index=False)

  def predict(self, x):
    xnew = sm.add_constant(x)
    if x.shape == xnew.shape:
      assert x.ndim == 2
      xnew = np.insert(x, 0, values=1, axis=1)
    return self._result.predict(xnew)
