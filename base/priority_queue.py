# 2019
# author: yuxuan

import heapq


class PriorityQueue(object):
  def __init__(self):
    self._q = []

  def add(self, k, v):
    heapq.heappush(self._q, (k, v))

  def pop(self):
    _, v = heapq.heappop(self._q)
    return v

  def size(self):
    return len(self._q)

  def empty(self):
    return self.size() == 0
