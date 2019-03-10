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


class BookFeature(Feature):
  def __init__(self, levels):
    self._book = None
    self._last_book = None
    self._levels = levels

  def on_feed(self, feed):
    if feed.feed_type == FeedType.BOOK:
      self._last_book = self._book
      self._book = feed

  def to_feature(self):
    features = []
    # Bid-ask spread
    features += [self._book.asks[0][0] / self._book.bids[0][0] - 1]

    # Bid-ask imbalance
    price_ratio = []
    qty_ratio = []
    base_price = self._book.bids[0][0]
    base_qty = self._book.bids[0][1]
    for i in range(self._levels):
      price_ratio.append(self._book.bids[i][0] / base_price)
      price_ratio.append(self._book.asks[i][0] / base_price)
      qty_ratio.append(self._book.bids[i][1] / base_qty)
      qty_ratio.append(self._book.asks[i][1] / base_qty)
    features += np.log(price_ratio[1:] + qty_ratio[1:]).tolist()

    # Change of best levels
    features += np.log(
        [self._book.bids[0][0] / self._last_book.bids[0][0],
         self._book.asks[0][0] / self._last_book.asks[0][0]]).tolist()
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
      v = 2 * total / qty / (self._book.bids[0][0] + self._book.asks[0][0])
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
      qty_imb = np.log(total_buys / total_sells)
      cnt_imb = np.log(count_buys / count_sells)
      return [qty_imb, cnt_imb]
    return [np.nan, np.nan]

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
