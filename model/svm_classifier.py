# 2019
# author: yuxuan

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


class SvmClassifier(object):
  def __init__(self):
    self._model = None

  def train_model(self, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    clf = svm.LinearSVC(loss='hinge', C=1, random_state = 42)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    accuracy = accuracy_score(y_train, predict)
    print("SVM training accuracy: " + "{0:.2f}".format(accuracy))
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print("SVM test accuracy: " + "{0:.2f}".format(accuracy))
    self._model = clf

  def predict(self, x):
    return self._model.predict(x)


if __name__ == '__main__':
  df = pd.read_csv('test3.csv')
  nan_mask = np.any(pd.isnull(df), 1)
  df = df[~nan_mask]
  columns = df.columns
  y_col = columns[-1]
  one_mask = np.abs(df[y_col]) > 8e-4
  df.loc[one_mask, y_col] = 1
  df.loc[~one_mask, y_col] = 0
  y = df.pop(y_col)
  model = SvmClassifier()
  model.train_model(df, y)
  assert False

  df = pd.read_csv('test3.csv')
  nan_mask = np.any(pd.isnull(df), 1)
  df = df[~nan_mask]
  columns = df.columns
  y_col = columns[-1]
  one_mask = np.abs(df[y_col]) > 8e-4
  df = df[one_mask]
  print(df.shape)
  df.loc[df[y_col] > 0, y_col] = 1
  df.loc[df[y_col] <= 0, y_col] = 0
  df.to_csv('up_down.csv', index=False)
  y = df.pop(y_col)
  model = SvmClassifier()
  model.train_model(df, y)
