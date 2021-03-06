# 2019
# author: yuxuan

import numpy as np
import pandas as pd
from absl import app, flags

from sampler.sampler import *
from feature.feature import *
from feeder.researcher import *
from feeder.feeder import base_quote_to_symbol


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'output_path',
    '',
    '')
flags.DEFINE_string(
    'save_path',
    '',
    '')
flags.DEFINE_string(
    'load_path',
    '',
    '')
flags.DEFINE_string(
    'os_path',
    '',
    '')


def add_samplers_features(researcher):
  researcher.subscribe_reference_product('Okex', 'ETH', 'USDT', '')

  #researcher.add_samplers(BestBookLevelTakenSampler(2))
  #researcher.add_samplers(FixedIntervalSampler(5))
  #researcher.add_samplers(PriceChangedSampler(10))
  #researcher.add_samplers(LargeTradeSampler(300, 0.5))
  researcher.add_samplers(FixedQuantitySampler(1000))

  researcher.add_feature(SnapshotBookFeature(3))
  researcher.add_feature(TimedVwapFeature(2))
  researcher.add_feature(TimedVwapFeature(5))
  researcher.add_feature(TimedVwapFeature(10))
  researcher.add_feature(TimedVwapFeature(30))
  researcher.add_feature(TimedVwapFeature(60))
  researcher.add_feature(ArFeature(StepBookFeature(), 3))
  researcher.add_feature(ArFeature(StepVwapFeature(), 3))
  researcher.add_feature(ArFeature(StepTradeFeature(), 3))
  researcher.add_feature(ArFeature(StepVolumeFeature(), 3))

  researcher.add_reference_product_feature(
      base_quote_to_symbol('ETH', 'USDT', ''),
      ReferenceProductFeature())


def extract_feature_reward(output_path):
  extractor = FeatureRewardExtractor(
      'Okex', 'ETH', 'USD', 20190329,
      #[20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127,
      # 20190128, 20190129, 20190130, 20190131, 20190201, 20190202, 20190203],
      [20190204, 20190205, 20190206, 20190207, 20190208, 20190209, 20190210],
      output_path)
  add_samplers_features(extractor)
  extractor.start()


def filter_nan(df):
  return df[~np.any(pd.isnull(df), 1)]


def method1(sample_path, classifier_cls):
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

  buy_enter_model = classifier_cls()
  buy_exit_model = classifier_cls()
  sell_enter_model = classifier_cls()
  sell_exit_model = classifier_cls()
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


def model_os(os_path, model, model_load_path, normalizer=None):
  model.load_model(model_load_path)
  df = pd.read_csv(os_path)
  df = filter_nan(df)
  y_col = df.columns[-1]
  if normalizer is not None:
    df = normalizer.normalize(df)
  y = df.pop(y_col)
  yhat = model.predict(df)
  sdf = pd.DataFrame({'y': y.tolist(), 'yhat': yhat.tolist()})
  print(np.corrcoef(y, yhat))
  sdf.to_csv('y_os.csv', index=False)


def method2(sample_path, model):
  df = pd.read_csv(sample_path)
  df = filter_nan(df)
  y_col = df.columns[-1]

  normalizer = None
  normalizer = Normalizer(df.describe().transpose())

  if FLAGS.os_path:
    model_os(FLAGS.os_path, model, FLAGS.load_path, normalizer)
    return

  df = normalizer.normalize(df)

  y = df.pop(y_col)
  print(y.describe())

  if FLAGS.load_path:
    model.load_model(FLAGS.load_path)
  else:
    model.train_model(df, y)
  y_hat = model.predict(df)
  y_hat = pd.DataFrame(y_hat)
  #print(df)
  print(y)
  print(y_hat)
  print(y_hat[0].value_counts())
  #y.to_csv('test_y.csv', index=False)
  #y_hat.to_csv('test_yhat.csv', index=False)
  corr = np.corrcoef(y.tolist(), y_hat.iloc[:, 0].tolist())
  print('overall corr:', corr)
  print(y_hat.describe())
  if FLAGS.save_path:
    model.save_model(FLAGS.save_path)

  class MySignals(Signals):
    def __init__(
        self, model,
        enter_buy_threshold, enter_sell_threshold,
        exit_buy_threshold, exit_sell_threshold,
        normalizer=None):
      self._model = model
      self._enter_buy_threshold = enter_buy_threshold
      self._enter_sell_threshold = enter_sell_threshold
      self._exit_buy_threshold = exit_buy_threshold
      self._exit_sell_threshold = exit_sell_threshold
      self._feature = None
      self._pred = 0.0
      self._normalizer = normalizer
      self._features = []
      self._yhat = []

    def on_feature(self, feature):
      if self._normalizer:
        nfeature = self._normalizer.normalize(feature)
      nfeature = np.array([nfeature])
      self._feature = nfeature
      if np.any(np.isnan(nfeature)):
        return
      self._pred = self._model.predict(nfeature)[0]
      self._features.append(feature)
      self._yhat.append(self._pred)

    def should_enter_buy(self):
      return self._pred > self._enter_buy_threshold

    def should_exit_buy(self):
      return self._pred < self._exit_buy_threshold

    def should_enter_sell(self):
      return self._pred < self._enter_sell_threshold

    def should_exit_sell(self):
      return self._pred > self._exit_sell_threshold

    def on_completed(self):
      df = pd.DataFrame(self._features)
      df['y'] = self._yhat
      df.to_csv('backtest_features.csv', index=False)

  enter_buy_threshold = np.percentile(y_hat, 95)
  enter_sell_threshold = np.percentile(y_hat, 5)
  exit_buy_threshold = np.percentile(y_hat, 5)
  exit_sell_threshold = np.percentile(y_hat, 95)
  print(enter_buy_threshold, exit_buy_threshold)
  print(enter_sell_threshold, exit_sell_threshold)
  #return

  backtest = BacktestReseacher(
    'Okex', 'ETH', 'USD', 20190329,
    [20190204, 20190205, 20190206, 20190207, 20190208, 20190209, 20190210],
    MySignals(
        model,
        enter_buy_threshold, enter_sell_threshold,
        exit_buy_threshold, exit_sell_threshold,
        normalizer))
  add_samplers_features(backtest)
  backtest.start()


def main(argv):
  assert FLAGS.output_path
  output_path = FLAGS.output_path
  #extract_feature_reward(output_path)
  #return
  from model.svm_classifier import SvmClassifier
  from model.sequential import SequentialClassifier, SequentialRegressor
  from model.linear_model import LinearModel
  #method1(output_path, SvmClassifier)
  #method1(output_path, SequentialClassifier)
  method2(output_path, SequentialRegressor(epochs=12, lr=0.001))
  #method2(output_path, LinearModel())


if __name__ == '__main__':
  app.run(main)
