# 2019
# author: yuxuan

import numpy as np
import pandas as pd
from absl import app, flags

from base.priority_queue import PriorityQueue
from feeder.feeder import Feeder
from feeder.feeder import FeedType
from sampler.sampler import *
from feature.feature import *


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'output_path',
    '',
    '')


def mid(book):
  return (book.bids[0][0] + book.asks[0][0]) / 2.0


class Researcher(object):
  def __init__(self, exchange, base, quote, expiry, dates, output_path):
    self._exchange = exchange
    self._base = base
    self._quote = quote
    self._dates = dates
    self._samplers = [BestBookLevelTakenSampler(3)]
    self._features = [
        BookFeature(3), TimedVwapFeature(5 * 60), TimedVwapFeature(30 * 60),
        StepTradeImbalanceFeature(), TimedVolumeFeature(3600, 5)]
    self._feeder = Feeder(dates, exchange, base, quote, expiry)
    self._feeder.subscribe_book(self.on_feed)
    self._feeder.subscribe_trade(self.on_feed)
    self._last_sampled_book = None
    self._current_book = None
    self._last_features = None
    self._reward = None
    self._feature_reward = []

  def output_feature_reward(self):
    df = pd.DataFrame(self._feature_reward)
    df.to_csv(FLAGS.output_path, index=False)

  def start(self):
    self._feeder.start_feed()
    self.output_feature_reward()

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
        self._feature_reward.append(self._last_features + [reward])
      features = []
      for feature in self._features:
        new_feature = feature.to_feature()
        assert isinstance(new_feature, list)
        features += new_feature
        feature.reset()
      self._last_features = features
      self._last_sampled_book = self._current_book


def main(argv):
  assert FLAGS.output_path
  researcher = Researcher(
      'Okex', 'ETH', 'USDT', None,
      [20190121, 20190122, 20190123, 20190124],
      FLAGS.output_path)
  researcher.start()


if __name__ == '__main__':
  app.run(main)
