import os
import glob

import numpy as np
import pandas as pd


def read_feed(dates, exchange, symbol, is_future=False):
  data = None
  for date in dates:
    dot_count = 3 + is_future
    dir_path = os.path.join('data', str(date))
    files = glob.glob('%s/%s*%s*' % (dir_path, symbol, exchange))
    files = [f for f in files if 'trade' not in f and f.count('.') == dot_count]
    assert len(files) == 1, '%s %s %s %s' % (date, exchange, symbol, files)
    df = pd.read_csv(files[0])
    data = df if data is None else data.append(df)
  data = data.reset_index(drop=True)
  return data


def volatility(date, exchange, symbol, is_future, interval):
  feed = read_feed(date, exchange, symbol, is_future)
  feed['mid'] = (feed['bid0'] + feed['ask0']) / 2.0
  feed['grp'] = (feed['timestamp'] / interval).astype(int)
  feed = feed.groupby('grp')['mid'].first()
  data = np.array(feed)
  price_change = np.log(data[1:] / data[:-1])
  return np.sqrt(np.mean(np.power(price_change, 2)))


def time_weighted_bid_ask_spread(date, exchange, symbol):
  df = read_feed(date, exchange, symbol)
  duration = df['timestamp'].diff()
  spread = 2.0 * (df['ask0'] - df['bid0']) / (df['ask0'] + df['bid0'])
  weighted_sum = np.sum(duration * spread)
  weighted_average = weighted_sum / np.sum(duration)
  return weighted_average


def main(dates):
  exchanges = ['Okex', 'Huobi', 'Upbit', 'Bithumb', 'Binance']
  quotes = ['USDT', 'USDT', 'KRW', 'KRW', 'USDT']
  bases = ['BTC', 'ETH', 'XRP', 'EOS']
  data = {exchange: {base: None for base in bases} for exchange in exchanges}
  vol = {exchange: {base: np.nan for base in bases} for exchange in exchanges}
  for exchange, quote in zip(exchanges, quotes):
    for base in bases:
      symbol = '%s-%s' % (base, quote)
      twa = time_weighted_bid_ask_spread(dates, exchange, symbol) * 1E4
      a_vol = volatility(dates, exchange, symbol, False, 600 * 1e9)
      data[exchange][base] = twa
      vol[exchange][base] = a_vol
      print(exchange, symbol, twa, a_vol)
  df = pd.DataFrame(data).transpose()
  print(df)
  vdf = pd.DataFrame(vol).transpose()
  print(vdf)


if __name__ == '__main__':
  main([20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127])
