# 2019
# author: yuxuan

import numpy as np
import pandas as pd

import statsmodels.api as sm


def regression(file_path):
  df = pd.read_csv(file_path)
  has_nan = np.any(pd.isnull(df), 1)
  df = df[~has_nan]
  y_idx = len(df.columns) - 1
  y_df = df.pop(str(y_idx))
  x_df = df
  columns = x_df.columns
  Xs = [x_df[col].tolist() for col in columns]
  Xs = np.array(Xs).T
  #Xs = sm.add_constant(Xs)
  Y = y_df.tolist()

  model = sm.OLS(Y, Xs)
  result = model.fit()
  print(result.summary())
  return result, Xs, Y


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


if __name__ == '__main__':
  regression()
