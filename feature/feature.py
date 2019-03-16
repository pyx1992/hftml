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

  def to_feature(self):
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

    # Current mid price
    features += [np.log((self._book.bids[0][0] + self._book.asks[0][0]) / 2.0)]

    # Volume weighted mid price
    a, b = 0.0, 0.0
    for i in range(self._levels):
      a += 1.0 * self._book.bids[i][0] / self._book.bids[i][1] + \
          1.0 * self._book.asks[i][0] / self._book.asks[i][1]
      b += 1.0 / self._book.bids[i][1] + 1.0 / self._book.asks[i][1]
      features += [np.log(a / b)]
    return features

  def reset(self):
    pass


class TimedBookFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.BOOK:
      self._dq.append(feed.timestamp, feed)

  def to_feature(self):
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
      twmid = np.nan
    features.append(np.log(twmid))

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
      twspread = np.nan
    features.append(twspread)

    # Time weighted best level qty.
    if bids0.shape[0] > 0:
      duration = bids0[-1,0] - bids0[0,0]
      if duration == 0:
        twbidq = bids0[0,2]
        twaskq = asks0[0,2]
      else:
        twbidq = np.sum(bids0[0,2] * (bids0[1:,0] - bids0[:-1,0]) / duration)
        twaskq = np.sum(asks0[0,2] * (asks0[1:,0] - asks0[:-1,0]) / duration)
    else:
      twbidq = np.nan
      twaskq = np.nan
    features += [np.log(twbidq), np.log(twaskq)]

    return features

  def reset(self):
    pass


class StepBookFeature(TimedBookFeature):
  def __init__(self):
    TimedBookFeature.__init__(self, -1)

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

  def to_feature(self):
    if not self._dq.empty():
      dq = self._dq.data()
      total = 0.0
      qty = 0.0
      for e in dq:
        total += e[1][0] * e[1][1]
        qty += e[1][1]
      v = total / qty
      return [np.log(v)]
    return [np.nan]

  def reset(self):
    pass


class StepVwapFeature(TimedVwapFeature):
  def __init__(self):
    TimedVwapFeature.__init__(self, -1)

  def reset(self):
    self._dq.clear()


class TimedTradeFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, (feed.price, feed.qty, feed.side))

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
    features += [np.log(total_buys), np.log(total_sells),
                 np.log(count_buys), np.log(count_sells)]

    return features

  def reset(self):
    pass


class StepTradeFeature(TimedTradeFeature):
  def __init__(self):
    TimedTradeFeature.__init__(self, -1)

  def reset(self):
    self._dq.clear()


class TimedVolumeFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, feed.qty)

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

  def reset(self):
    self._dq.clear()
