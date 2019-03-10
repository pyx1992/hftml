# 2019
# author: yuxuan

import os
from enum import Enum

from absl import flags

from base.priority_queue import PriorityQueue


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'feed_dir',
    'data',
    '')
flags.DEFINE_integer(
    'book_levels',
    3,
    '')


def symbol_to_base_quote(symbol):
  if '.' in symbol:
    symbol, expiry = symbol.split('.')
    base, quote = symbol.split('-')
    return base, quote, expiry
  else:
    base, quote = symbol.split('-')
    return base, quote


def base_quote_to_symbol(base, quote, expiry=None):
  if expiry:
    return '%s-%s.%s' % (base, quote, expiry)
  else:
    return '%s-%s' % (base, quote)


class OrderSide(Enum):
  BUY = 1
  SELL = 2

class FeedType(Enum):
  BOOK = 1
  TRADE = 2


class Feed(object):
  def __init__(self):
    self.timestamp = None
    self.exchange = None
    self.symbol = None


class Book(Feed):
  def __init__(self):
    self.bids = []
    self.asks = []
    self.feed_type = FeedType.BOOK


class Trade(Feed):
  def __init__(self):
    self.side = None
    self.price = None
    self.qty = None
    self.feed_type = FeedType.TRADE


class FeedFactory(object):
  def __init__(self, book_levels):
    self._book_levels = book_levels

  def create_feed(self, exchange, symbol, feed_type, text):
    if feed_type == FeedType.BOOK:
      feed = Book()
      data = text.split(',')
      feed.timestamp = int(data[0])
      for i in range(self._book_levels):
        bid_price = float(data[3 + 4 * i])
        bid_qty = float(data[3 + 4 * i + 1])
        ask_price = float(data[3 + 4 * i + 2])
        ask_qty = float(data[3 + 4 * i + 3])
        feed.bids.append((bid_price, bid_qty))
        feed.asks.append((ask_price, ask_qty))
    elif feed_type  == FeedType.TRADE:
      feed = Trade()
      data = text.split(',')
      feed.timestamp = int(data[0])
      if data[3] == 'B':
        order_side = OrderSide.BUY
      elif data[3] == 'S':
        order_side = OrderSide.SELL
      else:
        assert False, data[3]
      feed.side = order_side
      feed.price = float(data[4])
      feed.qty = float(data[5])
    else:
      assert False, feed_type
    feed.exchange = exchange
    feed.symbol = symbol
    return feed


class FeedReader(object):
  def __init__(self, date, exchange, base, quote, expiry, feed_type):
    self._date = date
    self._exchange = exchange
    self._base = base
    self._quote = quote
    self._symbol = base_quote_to_symbol(base, quote, expiry)
    self._expiry = expiry
    self._feed_type = feed_type
    self._file_path = self._get_feed_file_path()
    assert os.path.exists(self._file_path), self._file_path
    self._file = open(self._file_path, 'r')
    # Skip header.
    self._file.readline()
    self._feed_factory = FeedFactory(FLAGS.book_levels)

  @staticmethod
  def get_feed_file_path(data_dir, date, exchange, base, quote, expiry, feed_type):
    if expiry:
      file_path = '%s-%s.%s.%s.%s%s' % (
          base, quote, expiry, exchange, base, str(expiry)[-4:])
    else:
      file_path = '%s-%s.%s.%s_%s' % (
          base, quote, exchange, base.lower(), quote.lower())
    if feed_type == FeedType.BOOK:
      file_path = '%s.csv' % file_path
    elif feed_type == FeedType.TRADE:
      file_path = '%s_trade.csv' % file_path
    return os.path.join(data_dir, str(date), file_path)

  def _get_feed_file_path(self):
    return FeedReader.get_feed_file_path(
        FLAGS.feed_dir, self._date, self._exchange, self._base, self._quote,
        self._expiry, self._feed_type)

  def next(self):
    line = self._file.readline()
    if not line:
      self._file.close()
      return None
    return self._feed_factory.create_feed(
        self._exchange, self._symbol, self._feed_type, line.strip())


class Feeder(object):
  def __init__(self, dates, exchange, base, quote, expiry):
    self._dates = dates
    self._exchange = exchange
    self._base = base
    self._quote = quote
    self._symbol = base_quote_to_symbol(base, quote, expiry)
    self._expiry = expiry
    self._feed_readers = dict()
    self._feed = PriorityQueue()
    self._book_callback = None
    self._trade_callback = None

  def start_feed(self):
    while True:
      feed = self._next()
      if not feed:
        break
      if feed.feed_type == FeedType.BOOK and self._book_callback:
        self._book_callback(feed)
      elif feed.feed_type == FeedType.TRADE and self._trade_callback:
        self._trade_callback(feed)

  def _next(self):
    if self._feed.empty():
      return None
    key, feed = self._feed.pop()
    next_feed = self._feed_readers[key].next()
    if next_feed:
      self._feed.add(next_feed.timestamp, (key, next_feed))
    return feed

  def subscribe_book(self, func):
    self._book_callback = func
    for date in self._dates:
      key = (date, self._symbol, FeedType.BOOK)
      feed_reader = FeedReader(
          date, self._exchange, self._base, self._quote, self._expiry, FeedType.BOOK)
      feed = feed_reader.next()
      if feed:
        self._feed_readers[key] = feed_reader
        self._feed.add(feed.timestamp, (key, feed))

  def subscribe_trade(self, func):
    self._trade_callback = func
    for date in self._dates:
      key = (date, self._symbol, FeedType.TRADE)
      feed_reader = FeedReader(
          date, self._exchange, self._base, self._quote, self._expiry, FeedType.TRADE)
      feed = feed_reader.next()
      if feed:
        self._feed_readers[key] = feed_reader
        self._feed.add(feed.timestamp, (key, feed))
