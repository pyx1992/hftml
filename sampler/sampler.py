# 2019
# author: yuxuan

import numpy as np

from feeder.feeder import FeedType
from base.timed_deque import TimedDeque


class Sampler(object):
  def sampled(self, feed):
    raise NotImplementedError()


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


class LargeTradeSampler(Sampler):
  def __init__(self, timewindowsec, std_threshold):
    self._timewindow = timewindowsec * 10 ** 9
    self._threshold = std_threshold
    self._dq = TimedDeque(self._timewindow)

  def ready(self, feed):
    return self._dq.ready()

  def _triggered(self, trade):
    dq = self._dq.data()
    data = np.array(dq)
    mean = np.mean(data[:, 1])
    std = np.std(data[:, 1])
    return trade.qty > mean + self._threshold * std

  def sampled(self, feed):
    timestamp = feed.timestamp
    self._dq.upadte(timestamp)
    sampled = False
    if feed.feed_type == FeedType.TRADE:
      if self.ready(feed):
        sampled = self._triggered(feed)
      self._dq.append(timestamp, feed.qty)
    return sampled
