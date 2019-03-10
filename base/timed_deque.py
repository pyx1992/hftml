# 2019
# author: yuxuan

from collections import deque


class TimedDeque(object):
  def __init__(self, timewindow):
    self._timewindow = timewindow
    self._dq = deque()

  def ready(self, timestamp, multiplier=0.7):
    if len(self._dq) == 0:
      return False
    return timestamp - self._dq[0][0] > multiplier * self._timewindow

  def update(self, timestamp):
    if timestamp > 0:
      while len(self._dq) > 0 and self._dq[0][0] + self._timewindow < timestamp:
        self._dq.popleft()

  def append(self, timestamp, data):
    self.update(timestamp)
    self._dq.append((timestamp, data))

  def data(self):
    return self._dq

  def empty(self):
    return len(self._dq) == 0

  def clear(self):
    self._dq = deque()
