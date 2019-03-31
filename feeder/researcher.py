# 2019
# author: yuxuan

import datetime

import numpy as np
import pandas as pd

from feeder.feeder import Feeder, FeedType


def mid(book):
  return (book.bids[0][0] + book.asks[0][0]) / 2.0


class FeatureRewardResearcher(object):
  def __init__(self, exchange, base, quote, expiry, dates):
    self._exchange = exchange
    self._base = base
    self._quote = quote
    self._dates = dates
    self._samplers = []
    self._features = []
    self._feeder = Feeder(dates, exchange, base, quote, expiry)
    self._feeder.subscribe_book(self.on_feed)
    self._feeder.subscribe_trade(self.on_feed)
    self._last_sampled_book = None
    self._current_book = None
    self._last_features = None
    self._last_sampled_ts = np.nan
    self._columns = ['interval length']

  def add_samplers(self, sampler):
    self._samplers.append(sampler)

  def add_feature(self, feature):
    self._features.append(feature)

  def on_sampled(self, feature, reward):
    raise NotImplementedError()

  def on_completed(self):
    raise NotImplementedError()

  def start(self):
    for f in self._features:
      self._columns += f.feature_names()
    self._feeder.start_feed()
    self.on_completed()

  def ready(self):
    return self._last_sampled_book is not None and self._last_features is not None

  def get_reward(self):
    return np.log(mid(self._current_book) / mid(self._last_sampled_book))

  def on_feed(self, feed):
    if feed.feed_type == FeedType.BOOK:
      self._current_book = feed

    for feature in self._features:
      feature.on_feed(feed)
    sampled = False
    for sampler in self._samplers:
      if sampler.sampled(feed):
        sampled = True
        break
    if sampled:
      if self.ready():
        reward = self.get_reward()
        if feed.timestamp == self._last_sampled_ts:
          return
        self.on_sampled(self._last_features, reward)
      features = [np.log(feed.timestamp - self._last_sampled_ts + 0.1)]
      for feature in self._features:
        new_feature = feature.to_feature()
        assert isinstance(new_feature, list)
        features += new_feature
        feature.reset()
      self._last_features = features
      self._last_sampled_book = self._current_book
      self._last_sampled_ts = feed.timestamp
      for sampler in self._samplers:
        sampler.other_sampled()


class FeatureRewardExtractor(FeatureRewardResearcher):
  def __init__(self, exchange, base, quote, expiry, dates, output_path):
    FeatureRewardResearcher.__init__(self, exchange, base, quote, expiry, dates)
    self._output_path = output_path
    self._feature_reward = []

  def on_sampled(self, feature, reward):
    self._feature_reward.append(feature + [reward])

  def on_completed(self):
    df = pd.DataFrame(self._feature_reward)
    columns = self._columns + ['y']
    df.to_csv(self._output_path, index=False, header=columns)


class Signals(object):
  def should_enter_buy(self):
    raise NotImplementedError()

  def should_enter_sell(self):
    raise NotImplementedError()

  def should_exit_buy(self):
    raise NotImplementedError()

  def should_exit_sell(self):
    raise NotImplementedError()


class BacktestReseacher(FeatureRewardResearcher):
  def __init__(self, exchange, base, quote, expiry, dates, signals):
    FeatureRewardResearcher.__init__(self, exchange, base, quote, expiry, dates)
    self._signals = signals
    self._position = 0.0
    self._cash = 0.0
    self._volume = 0.0
    self._pnl = 0.0
    self._ts = []
    self._pnls = []
    self._volumes = []

  def on_sampled(self, feature, _):
    self._signals.on_feature(feature)
    target_pos = self._position
    if self._signals.should_enter_buy():
      target_pos = 1
    elif self._signals.should_enter_sell():
      target_pos = -1
    else:
      if self._position > 0 and self._signals.should_exit_buy():
        target_pos = 0
      elif self._position < 0 and self._signals.should_exit_sell():
        target_pos = 0

    diff_pos = target_pos - self._position
    dt = datetime.datetime.fromtimestamp(self._current_book.timestamp * 1e-9)
    price = mid(self._current_book)
    if abs(diff_pos) > 1e-6:
      if diff_pos > 0:
        #price = self._current_book.asks[0][0]
        print('Buy %f @ %f %s' % (abs(diff_pos), price, dt))
      else:
        #price = self._current_book.bids[0][0]
        print('Sell %f @ %f %s' % (abs(diff_pos), price, dt))
      diff_cash = -diff_pos * price
      self._cash += diff_cash
      self._position = target_pos
      self._volume += abs(diff_cash)
      self._pnl = self._cash + self._position * (
          self._current_book.bids[0][0] + self._current_book.asks[0][0]) / 2.0
      self._ts.append(self._current_book.timestamp)
      self._pnls.append(self._pnl)
      self._volumes.append(self._volume)
      print('Pnl: %f, Volume: %f' % (self._pnl, self._volume))

  def on_completed(self):
    print('Total Pnl: %f' % self._pnl)
    print('Total Volume %f' % self._volume)
    data = pd.DataFrame({
      'timestamp': self._ts,
      'pnl': self._pnls,
      'volumes': self._volumes})
    data.to_csv('sim_result.csv', index=False)
