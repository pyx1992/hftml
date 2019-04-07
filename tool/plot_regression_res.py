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


def plot_regression(train_path, test_path):
  fig, axes = plt.subplots(nrows=2, ncols=1)
  train_df = pd.read_csv(train_path)
  train_df.set_index('y').plot(
      style='.', ax=axes[0], xlim=(-5,5), ylim=(-5,5))
  test_df = pd.read_csv(test_path)
  test_df.set_index('y').plot(
      style='.', ax=axes[1], xlim=(-5,5), ylim=(-5,5))
  plt.show()


def main(argv):
  plot_regression(FLAGS.train_path, FLAGS.test_path)


if __name__ == '__main__':
  app.run(main)
