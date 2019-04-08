import numpy as np
import pandas as pd
from absl import app, flags
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS


flags.DEFINE_string(
    'train_path',
    '',
    '')
flags.DEFINE_string(
    'test_path',
    '',
    '')
flags.DEFINE_string(
    'os_path',
    '',
    '')


def plot_regression(train_path, test_path, os_path=None):
  ncols = 2
  if os_path is not None:
    ncols = 3
  fig, axes = plt.subplots(nrows=1, ncols=ncols)
  train_df = pd.read_csv(train_path)
  train_df.set_index('y').plot(
      style='.', ax=axes[0], xlim=(-5,5), ylim=(-5,5))
  test_df = pd.read_csv(test_path)
  test_df.set_index('y').plot(
      style='.', ax=axes[1], xlim=(-5,5), ylim=(-5,5))
  if os_path:
    os_df = pd.read_csv(os_path)
    os_df.set_index('y').plot(
        style='.', ax=axes[2], xlim=(-5,5), ylim=(-5,5))
  plt.show()


def main(argv):
  plot_regression(FLAGS.train_path, FLAGS.test_path, FLAGS.os_path)


if __name__ == '__main__':
  app.run(main)
