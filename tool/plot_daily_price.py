# 2019
# author: yuxuan

import pandas as pd
import matplotlib.pyplot as plt
from absl import app

from feeder.feeder import FeedReader
from feeder.feeder import FeedType


def main(argv):
  file_path = FeedReader.get_feed_file_path(
      'data', 20190129, 'Okex', 'ETH', 'USDT', None, FeedType.BOOK)
  df = pd.read_csv(file_path)
  df['datetime'] = pd.to_datetime(df['timestamp'])
  df = df.set_index('datetime')
  df['mid'] = (df['bid0'] + df['ask0']) / 2.0
  df['mid'].plot()
  plt.show()


if __name__ == '__main__':
  app.run(main)
