# 2019
# author: yuxuan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot():
  pl = pd.read_csv('sim_result.csv')
  pl['datetime'] = pd.to_datetime(pl['timestamp'])
  pl = pl.set_index('datetime')
  fig, ax = plt.subplots()
  ax3 = ax.twinx()
  ax.plot(pl[['pnl']])
  ax3.plot(pl[['volumes']], color='r')
  ax3.legend([ax.get_lines()[0], ax3.get_lines()[0]], ['Pnl', 'Volume'])
  plt.show()


if __name__ == '__main__':
  plot()
