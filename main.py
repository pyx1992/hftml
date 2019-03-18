# 2019
# author: yuxuan

import numpy as np
import pandas as pd
from absl import app, flags

from sampler.sampler import *
from feature.feature import *
from feeder.researcher import *


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'output_path',
    '',
    '')


def add_samplers_features(researcher):
  #researcher.add_samplers(BestBookLevelTakenSampler(2))
  #researcher.add_samplers(FixedIntervalSampler(5))
  #researcher.add_samplers(PriceChangedSampler(5))
  researcher.add_samplers(LargeTradeSampler(300, 2))

  researcher.add_feature(SnapshotBookFeature(2))
  researcher.add_feature(TimedVwapFeature(1 * 60))
  researcher.add_feature(TimedVwapFeature(2 * 60))
  researcher.add_feature(TimedVwapFeature(5 * 60))
  researcher.add_feature(TimedVwapFeature(10 * 60))
  researcher.add_feature(ArFeature(StepBookFeature(), 3))
  researcher.add_feature(ArFeature(StepVwapFeature(), 3))
  researcher.add_feature(ArFeature(StepTradeFeature(), 2))
  researcher.add_feature(ArFeature(StepVolumeFeature(), 2))


def extract_feature_reward(output_path):
  extractor = FeatureRewardExtractor(
      'Okex', 'ETH', 'USD', 20190329,
      [20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127],
      output_path)
  add_samplers_features(extractor)
  extractor.start()


def filter_nan(df):
  return df[~np.any(pd.isnull(df), 1)]


def method1(sample_path, model_cls):
  enter_threshold = 5e-4
  exit_threshold = 2e-4
  df = pd.read_csv(sample_path)
  df = filter_nan(df)
  y_col = df.columns[-1]
  y = df.pop(y_col)

  def train_model(model, x, y):
    print(y.value_counts())
    one_mask = y == 1
    zero_mask = y == 0
    multiplier = np.sum(zero_mask) / np.sum(one_mask)
    if multiplier >= 2:
      multiplier = int(multiplier)
      x1 = x[one_mask]
      y1 = y[one_mask]
      for i in range(multiplier // 2):
        x = x.append(x1)
        y = y.append(y1)
    print(y.value_counts())
    model.train_model(x, y)

  buy_enter_model = model_cls()
  buy_exit_model = model_cls()
  sell_enter_model = model_cls()
  sell_exit_model = model_cls()
  train_model(buy_enter_model, df, y > enter_threshold)
  train_model(buy_exit_model, df, y < -exit_threshold)
  train_model(sell_enter_model, df, y < -enter_threshold)
  train_model(sell_exit_model, df, y > exit_threshold)
  #return

  class MySignals(Signals):
    def __init__(self,
                 enter_buy_model, exit_buy_model,
                 enter_sell_model, exit_sell_model):
      self._enter_buy_model = enter_buy_model
      self._exit_buy_model = exit_buy_model
      self._enter_sell_model = enter_sell_model
      self._exit_sell_model = exit_sell_model
      self._feature = None
      self._should_enter_buy = None
      self._should_exit_buy = None
      self._should_enter_sell = None
      self._should_exit_sell = None

    def on_feature(self, feature):
      self._feature = feature
      if np.any(np.isnan(feature)):
        return
      feature = [feature]
      self._should_enter_buy = self._enter_buy_model.predict(feature)
      self._should_exit_buy = self._exit_buy_model.predict(feature)
      self._should_enter_sell = self._enter_sell_model.predict(feature)
      self._should_exit_sell = self._exit_sell_model.predict(feature)

    def should_enter_buy(self):
      return self._should_enter_buy and not self._should_exit_buy and \
          not self._should_enter_sell

    def should_exit_buy(self):
      return self._should_exit_buy or self._should_enter_sell

    def should_enter_sell(self):
      return self._should_enter_sell and not self._should_exit_sell and \
          not self._should_enter_buy

    def should_exit_sell(self):
      return self._should_exit_sell or self._should_enter_buy


  backtest = BacktestReseacher(
    'Okex', 'ETH', 'USD', 20190329,
    [20190128, 20190129],
    MySignals(buy_enter_model, buy_exit_model, sell_enter_model, sell_exit_model))
  add_samplers_features(backtest)
  backtest.start()


def main(argv):
  assert FLAGS.output_path
  output_path = FLAGS.output_path
  extract_feature_reward(output_path)
  #return
  from model.svm_classifier import SvmClassifier
  from model.sequential import SequentialClassifier
  #method1(output_path, SvmClassifier)
  method1(output_path, SequentialClassifier)
  return


  from model.linear_model import regression, LinearModelSignals
  olsres, Xs, Y = regression(FLAGS.output_path)
  predict_y = olsres.predict(Xs)
  enter_buy = np.percentile(predict_y, 99)
  enter_sell = np.percentile(predict_y, 1)
  exit_buy = np.percentile(predict_y, 30)
  exit_sell = np.percentile(predict_y, 70)
  print('enter buy:  %f' % enter_buy)
  print('enter sell: %f' % enter_sell)
  print('exit buy:   %f' % exit_buy)
  print('exit sell:  %f' % exit_sell)
  #return
  lm_signals = LinearModelSignals(
      olsres, enter_buy, enter_sell, exit_buy, exit_sell)
  backtest = BacktestReseacher(
      'Okex', 'ETH', 'USD', 20190329,
      [20190128],
      lm_signals)
  add_samplers_features(backtest)
  backtest.start()


if __name__ == '__main__':
  app.run(main)
