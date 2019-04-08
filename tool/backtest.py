# 2019
# author: yuxuan

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def sim():
  orig_features = pd.read_csv('nnfq1000.csv').dropna()
  stat = orig_features.describe().transpose()
  print(stat)

  bt_df = pd.read_csv('backtest_features.csv')
  bt_df.columns = orig_features.columns
  os_df = pd.read_csv('nnfq1000_os.csv').dropna().reset_index(drop=True)
  ychange = os_df['y'].copy()
  t_os_df = (os_df - stat['mean']) / stat['std']
  print(bt_df.tail())
  print(t_os_df.tail())

  y_train = pd.read_csv('y_train.csv')
  y_test = pd.read_csv('y_test.csv')
  yhat = np.array(y_train['yhat'].tolist() + y_test['yhat'].tolist())
  enter_buy = np.percentile(yhat, 95)
  enter_sell = np.percentile(yhat, 5)
  iis = []
  pnls = []
  vols = []
  pos = 0
  pnl = 0
  vol = 0
  for i in range(bt_df.shape[0]):
    ypred = bt_df.iloc[i]['y']
    change = ychange[i]
    new_pos = pos
    #if pos > 0 and ypred < exit_buy:
    #  new_pos = 0
    #elif pos < 0 and ypred > exit_sell:
    #  new_pos = 0
    if ypred > enter_buy:
      new_pos = 1
    elif ypred < enter_sell:
      new_pos = -1
    pnl += pos * change
    vol += abs(new_pos - pos)
    if pos != new_pos:
      pos = new_pos
      print(i, pnl, vol, pos)
      iis.append(i)
      pnls.append(pnl)
      vols.append(vol)
  res = pd.DataFrame({'i': iis, 'pnl': pnls, 'vol': vol})
  res.set_index('i')['pnl'].plot()
  plt.show()



if __name__ == '__main__':
  sim()
