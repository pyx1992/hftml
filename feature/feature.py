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


class BookFeature(Feature):
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
    features += [self._get_delta_book_imbalance()]

    # Current mid price
    features += [np.log((self._book.bids[0][0] + self._book.asks[0][0]) / 2.0)]
    return features

  def reset(self):
    pass


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


class TimedTradeImbalanceFeature(Feature):
  def __init__(self, timewindowsec):
    self._timewindow = timewindowsec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, (feed.price, feed.qty, feed.side))

  def to_feature(self):
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
      qty_imb = total_buys / total_sells
      cnt_imb = count_buys / count_sells
      return [qty_imb, cnt_imb]
    return [1, 1]

  def reset(self):
    pass


class StepTradeImbalanceFeature(TimedTradeImbalanceFeature):
  def __init__(self):
    TimedTradeImbalanceFeature.__init__(self, -1)

  def reset(self):
    self._dq.clear()


class TimedVolumeFeature(Feature):
  def __init__(self, timewindowsec, measure_window_sec):
    self._timewindow = timewindowsec * 10 ** 9
    self._measure_window = measure_window_sec * 10 ** 9
    self._dq = TimedDeque(self._timewindow)

  def on_feed(self, feed):
    if feed.feed_type == FeedType.TRADE:
      self._dq.append(feed.timestamp, feed.qty)

  def to_feature(self):
    data = pd.DataFrame(np.array(self._dq.data()))
    if data.shape[0] > 0:
      data.columns = ['timestamp', 'qty']
      data['group'] = (
          data.iloc[-1]['timestamp'] - data['timestamp']) / self._measure_window
      data['group'] = data['group'].astype(int)
      gqty = data.groupby('group')['qty'].sum()
      total_intervals = max(gqty.index) + 1
      no_trades_intervals = total_intervals - gqty.shape[0]
      gqty = [0] * no_trades_intervals + gqty.tolist()
      assert len(gqty) == total_intervals, '%s %s' % (len(gqty), total_intervals)
      stats = (gqty[-1] - np.mean(gqty)) / np.std(gqty)
      return [stats]
    return [np.nan]

  def reset(self):
    pass
