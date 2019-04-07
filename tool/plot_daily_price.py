# 2019
# author: yuxuan

import pandas as pd
import matplotlib.pyplot as plt
from absl import app

from feeder.feeder import FeedReader
from feeder.feeder import FeedType


def plot_all(_):
  data = None
  vpoint = None
  for date in [
      20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127,
      20190128, 20190129, 20190130, 20190131, 20190201, 20190202, 20190203]:
    file_path = FeedReader.get_feed_file_path(
      'data', date, 'Okex', 'ETH', 'USD', 20190329, FeedType.BOOK)
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['mid'] = (df['bid0'] + df['ask0']) / 2.0
    df = df[['datetime', 'mid']]
    data = df if data is None else data.append(df)
    if date == 20190131:
      vpoint = df.iloc[0]['datetime']
  data = data.reset_index(drop=True)
  data.set_index('datetime').plot()
  plt.axvline(x=vpoint, color='r', linestyle='--', lw=2)
  plt.show()


def main(argv):
  file_path = FeedReader.get_feed_file_path(
      'data', 20190129, 'Okex', 'ETH', 'USD', 20190329, FeedType.BOOK)
  df = pd.read_csv(file_path)
  df['datetime'] = pd.to_datetime(df['timestamp'])
  df = df.set_index('datetime')
  df['mid'] = (df['bid0'] + df['ask0']) / 2.0
  df['mid'].plot()
  plt.show()


if __name__ == '__main__':
  app.run(plot_all)
