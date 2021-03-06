# 2019
# author: yuxuan

import numpy as np

from feeder.feeder import FeedType
from base.timed_deque import TimedDeque


class Sampler(object):
  def sampled(self, feed):
    raise NotImplementedError()

  def other_sampled(self):
    pass


# Sampled if best levels of book are taken.
class BestBookLevelTakenSampler(Sampler):
  def __init__(self, levels):
    assert levels > 0
    self._levels = levels
    self._last_book = None

  def sampled(self, feed):
    sampled = False
    if feed.feed_type == FeedType.BOOK:
      if self._last_book:
        if self._last_book.bids[self._levels - 1][0] > feed.bids[0][0]:
          sampled = True
        if self._last_book.asks[self._levels - 1][0] < feed.asks[0][0]:
          sampled = True
      self._last_book = feed
    return sampled


# Sampled if trade with large volume occurs.
class LargeTradeSampler(Sampler):
  def __init__(self, timewindowsec, std_threshold):
    self._timewindow = timewindowsec * 10 ** 9
    self._threshold = std_threshold
    self._dq = TimedDeque(self._timewindow)

  def ready(self, feed):
    return self._dq.ready(feed.timestamp)

  def _triggered(self, trade):
    dq = self._dq.data()
    data = np.array(dq)
    mean = np.mean(data[:, 1])
    std = np.std(data[:, 1])
    return trade.qty > mean + self._threshold * std

  def sampled(self, feed):
    timestamp = feed.timestamp
    self._dq.update(timestamp)
    sampled = False
    if feed.feed_type == FeedType.TRADE:
      if self.ready(feed):
        sampled = self._triggered(feed)
      self._dq.append(timestamp, feed.qty)
    return sampled


# Sampled every fixed interval.
class FixedIntervalSampler(Sampler):
  def __init__(self, min_intervalsec):
    self._min_interval = min_intervalsec * 10 ** 9
    self._last_sampled_ts = 0
    self._last_feed = None

  def sampled(self, feed):
    sampled = False
    if feed.timestamp - self._last_sampled_ts >= self._min_interval:
      self._last_sampled_ts = feed.timestamp
      sampled = True
    self._last_feed = feed
    return sampled

  def other_sampled(self):
    self._last_sampled_ts = self._last_feed.timestamp


# Sampled if price changed beyond threshold.
class PriceChangedSampler(Sampler):
  def __init__(self, threshold_bp):
    self._threshold = threshold_bp * 1e-4
    self._ref_mid_price = 0
    self._last_book = None

  def sampled(self, feed):
    sampled = False
    if feed.feed_type == FeedType.BOOK:
      mid = (feed.bids[0][0] + feed.asks[0][0]) / 2.0
      if abs(mid - self._ref_mid_price) > self._ref_mid_price * self._threshold:
        self._ref_mid_price = mid
        sampled = True
      self._last_book = feed
    return sampled

  def other_sampled(self):
    self._ref_mid_price = (
        self._last_book.bids[0][0] + self._last_book.asks[0][0]) / 2.0


class FixedQuantitySampler(Sampler):
  def __init__(self, qty):
    self._qty = qty
    self._acc_qty = 0.0

  def sampled(self, feed):
    sampled = False
    if feed.feed_type == FeedType.TRADE:
      self._acc_qty += feed.qty
      if self._acc_qty >= self._qty:
        sampled = True
    return sampled

  def other_sampled(self):
    self._acc_qty = 0.0
