# 2019
# author: yuxuan

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyAdaBoostClassifier(object):
  def __init__(self):
    self._model = None

  def train_model(self, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(random_state=42, max_depth=2),
        n_estimators=5, random_state=42, algorithm='SAMME')
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_train)
    accuracy = accuracy_score(y_train, predict)
    print("Boosted DT training accuracy: " + "{0:.2f}".format(accuracy))
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print("Boosted DT test accuracy: " + "{0:.2f}".format(accuracy))
    self._model = clf

  def predict(self, x):
    return self._model.predict(x)


if __name__ == '__main__':
  df = pd.read_csv('up_down.csv')
  y_col = df.columns[-1]
  y = df.pop(y_col)
  clf = MyAdaBoostClassifier()
  clf.train_model(df, y)
