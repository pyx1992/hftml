# 2019
# author: yuxuan

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tool.plot_pnl_volume import plot_pairs


def sim():
  orig_features = pd.read_csv('ref_fq1000.csv').dropna()
  stat = orig_features.describe().transpose()
  print(stat)

  y_os = pd.read_csv('y_os.csv')
  os_df = pd.read_csv('ref_fq1000_os.csv').dropna().reset_index(drop=True)
  ychange = os_df['y'].copy()
  t_os_df = (os_df - stat['mean']) / stat['std']
  print(y_os.shape, ychange.shape)
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
  for i in range(y_os.shape[0]):
    ypred = y_os.iloc[i]['yhat']
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
  res = pd.DataFrame({'i': iis, 'pnl': pnls, 'vol': vols})
  res = res.set_index('i')
  print(res)
  plot_pairs(res['pnl'], res['vol'])



if __name__ == '__main__':
  sim()
