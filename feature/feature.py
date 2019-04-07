# 2019
# author: yuxuan

from collections import deque

import numpy as np
import pandas as pd

from base.timed_deque import TimedDeque
from feeder.feeder import FeedType
from feeder.feeder import OrderSide
from util.math import sigmoid


class Feature(object):
  def feature_names(self):
    raise NotImplementedError()

  def on_feed(self, feed):
    raise NotImplementedError()

  def to_feature(self):
    raise NotImplementedError()

  def reset(self):
    raise NotImplementedError()


class ArFeature(Feature):
  def __init__(self, feature, ar_steps):
    self._feature = feature
    self._ar_steps = ar_steps
    self._dq = deque(maxlen=ar_steps + 1)

  def feature_names(self):
    names = []
    for i in range(self._ar_steps + 1):
      names += ['ar_%d %s' % (i, n) for n in self._feature.feature_names()]
    return names

  def on_feed(self, feed):
    self._feature.on_feed(feed)

  def to_feature(self, ar=10000):
    feature = self._feature.to_feature()
    features = list(feature)
    for i in range(min(ar, self._ar_steps)):
      idx = len(self._dq) - 1 - i
      if idx >= 0:
        features += self._dq[-(i + 1)]
      else:
        features += [np.nan] * len(feature)
    self._dq.append(feature)
    return features

  def reset(self):
    self._feature.reset()

  def ready(self):
    return len(self._dq) == self._ar_steps + 1


class SnapshotBookFeature(Feature):
  def __init__(self, levels):
    self._book = None
    self._last_book = None
    self._levels = levels

  def on_feed(self, feed):
    if feed.feed_type == FeedType.BOOK:
      self._last_book = self._book
      self._book = feed

  def _get_delta_book_imbalance(self):
    if self._last_book is None:
      return 0
    old_bid = 0
    for i in range(self._levels):
      old_bid += self._last_book.bids[i][1]
    new_bid = 0
    for bid in self._book.bids:
      if bid[0] < self._last_book.bids[self._levels - 1][0]:
        break
      new_bid += bid[1]
    delta_bid = new_bid - old_bid

    old_ask = 0
    for i in range(self._levels):
      old_ask += self._last_book.asks[i][1]
    new_ask = 0
    for ask in self._book.asks:
      if ask[0] > self._last_book.asks[self._levels - 1][0]:
        break
      new_ask += ask[1]
    delta_ask = new_ask - old_ask

    bs_overwhelm_ratio = (delta_bid - delta_ask) / (old_bid + old_ask)
    return bs_overwhelm_ratio

  def feature_names(self):
    names = []
    names.append('snb book spread')
    for i in range(self._levels):
      names.append('snb book imbalance %d' % (i + 1))
    for i in range(self._levels):
      names.append('snb vw mid price diff %d' % (i + 1))
    return names

  def to_feature(self):
    current_mid = (self._book.bids[0][0] + self._book.asks[0][0]) / 2.0
    log_current_mid = np.log(current_mid)

    features = []
    # Bid-ask spread
    features += [(self._book.asks[0][0] - self._book.bids[0][0]) /
                 (self._book.bids[0][0] + self._book.asks[0][0])]

    # Bid-ask imbalance
    bid_qty = []
    ask_qty = []
    for i in range(self._levels):
      bid_qty.append(self._book.bids[i][1])
      ask_qty.append(self._book.asks[i][1])
      total_bid = sum(bid_qty)
      total_ask = sum(ask_qty)
      features += [(total_bid - total_ask) / (total_bid + total_ask)]

    # Buy-sell overwhelming ratio
    #features += [self._get_delta_book_imbalance()]

    # Volume weighted mid price - current mid
    a, b = 0.0, 0.0
    for i in range(self._levels):
      wbid = 1.0 / self._book.bids[i][1]
      wask = 1.0 / self._book.asks[i][1]
      a += self._book.bids[i][0] * wbid + self._book.asks[i][0] * wask
      b += wbid + wask
      features += [np.log(a / b) - log_current_mid]
    return features

  def reset(self):
    pass


class TimedBookFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)
    self._last_book = None

  def on_feed(self, feed):
    if feed.feed_type == FeedType.BOOK:
      self._dq.append(feed.timestamp, feed)
      self._last_book = feed

  def feature_names(self):
    names = []
    names.append('tb tw mid price')
    names.append('tb tw book spread')
    names += ['tb tw best bid ask imbalance']
    return names

  def to_feature(self):
    current_mid = (self._last_book.bids[0][0] + self._last_book.asks[0][0]) / 2.0
    log_current_mid = np.log(current_mid)

    features = []
    bids0 = np.array([[data[0]] + list(data[1].bids[0]) for data in self._dq.data()])
    asks0 = np.array([[data[0]] + list(data[1].asks[0]) for data in self._dq.data()])

    # Time weighted mid price
    if bids0.shape[0] > 0:
      duration = bids0[-1,0] - bids0[0,0]
      if duration == 0:
        twmid = (bids0[0,1] + asks0[0,1]) / 2.0
      else:
        twmid = np.sum(
            (bids0[:-1,1] + asks0[:-1,1]) / 2.0 * (bids0[1:,0] - bids0[:-1,0])) \
            / duration
    else:
      twmid = (self._last_book.bids[0][0] + self._last_book.asks[0][0]) / 2.0
    features.append(np.log(twmid) - log_current_mid)

    # Time weighted bid ask spread.
    if bids0.shape[0] > 0:
      duration = bids0[-1,0] - bids0[0,0]
      if duration == 0:
        twspread = (asks0[0,1] - bids0[0,1]) / (bids0[0,1] + asks0[0,1])
      else:
        twspread = np.sum(
            (asks0[0,1] - bids0[0,1]) / (bids0[0,1] + asks0[0,1]) *
            (bids0[1:,0] - bids0[:-1,0])) / duration
    else:
      twspread = (self._last_book.asks[0][0] - self._last_book.bids[0][0]) / \
          (self._last_book.asks[0][0] + self._last_book.bids[0][0])
    features.append(twspread)

    # Time weighted best level imbalance.
    if bids0.shape[0] > 0:
      duration = bids0[-1,0] - bids0[0,0]
      if duration == 0:
        twbidq = bids0[0,2]
        twaskq = asks0[0,2]
      else:
        twbidq = np.sum(bids0[0,2] * (bids0[1:,0] - bids0[:-1,0]) / duration)
        twaskq = np.sum(asks0[0,2] * (asks0[1:,0] - asks0[:-1,0]) / duration)
    else:
      twbidq = self._last_book.bids[0][1]
      twaskq = self._last_book.asks[0][1]
    features += [(twbidq - twaskq) / (twbidq + twaskq)]

    return features

  def reset(self):
    pass


class StepBookFeature(TimedBookFeature):
  def __init__(self):
    TimedBookFeature.__init__(self, -1)

  def feature_names(self):
    return ['sb %s' % n.split(' ', 1)[1] for n in TimedBookFeature.feature_names(self)]

  def reset(self):
    self._dq.clear()


class TimedVwapFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)
    self._book = None

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, (feed.price, feed.qty))
    else:
      self._book = feed

  def feature_names(self):
    return ['tt vwap']

  def to_feature(self):
    current_mid = (self._book.bids[0][0] + self._book.asks[0][0]) / 2.0
    log_current_mid = np.log(current_mid)

    if not self._dq.empty():
      dq = self._dq.data()
      total = 0.0
      qty = 0.0
      for e in dq:
        total += e[1][0] * e[1][1]
        qty += e[1][1]
      v = total / qty
      return [np.log(v) - log_current_mid]
    return [np.nan]

  def reset(self):
    pass


class StepVwapFeature(TimedVwapFeature):
  def __init__(self):
    TimedVwapFeature.__init__(self, -1)

  def feature_names(self):
    return ['st %s' % n.split(' ', 1)[1] for n in TimedVwapFeature.feature_names(self)]

  def reset(self):
    self._dq.clear()


class TimedTradeFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, (feed.price, feed.qty, feed.side))

  def feature_names(self):
    names = []
    names += ['tt qty imbalance', 'tt cnt imbalance']
    return names

  def to_feature(self):
    features = []
    total_buys = 0.1
    count_buys = 0.1
    total_sells = 0.1
    count_sells = 0.1
    if not self._dq.empty():
      buys = np.array([[e[0], e[1]] for _, e in self._dq.data() if e[2] == OrderSide.BUY])
      sells = np.array([[e[0], e[1]] for _, e in self._dq.data() if e[2] == OrderSide.SELL])
      total_buys = 0.1
      count_buys = 0.1
      if buys.shape[0] > 0:
        total_buys += np.sum(buys[:, 1])
        count_buys += buys.shape[0]
      total_sells = 0.1
      count_sells = 0.1
      if sells.shape[0] > 0:
        total_sells += np.sum(sells[:, 1])
        count_sells += sells.shape[0]
    qty_imb = (total_buys - total_sells) / (total_buys + total_sells)
    cnt_imb = (count_buys - count_sells) / (count_buys + count_sells)
    features += [qty_imb, cnt_imb]
    return features

  def reset(self):
    pass


class StepTradeFeature(TimedTradeFeature):
  def __init__(self):
    TimedTradeFeature.__init__(self, -1)

  def feature_names(self):
    return ['st %s' % n.split(' ', 1)[1] for n in TimedTradeFeature.feature_names(self)]

  def reset(self):
    self._dq.clear()


class TimedVolumeFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, feed.qty)

  def feature_names(self):
    names = []
    names += ['tt mean trade qty', 'tt std trade qty']
    names.append('tt trade cnt')
    return names

  def to_feature(self):
    features = []
    data = np.array(self._dq.data())
    # Trade qty mean and std
    mean_qty = 0.1
    std_qty = 0.1
    if not self._dq.empty():
      mean_qty += np.mean(data[:,1])
      std_qty += np.std(data[:,1])
    features += [np.log(mean_qty), np.log(std_qty)]

    # Trade count
    features.append(np.log(self._dq.size() + 0.1))

    return features

  def reset(self):
    pass


class StepVolumeFeature(TimedVolumeFeature):
  def __init__(self):
    TimedVolumeFeature.__init__(self, -1)

  def feature_names(self):
    return ['st %s' % n.split(' ', 1)[1] for n in TimedVolumeFeature.feature_names(self)]

  def reset(self):
    self._dq.clear()
