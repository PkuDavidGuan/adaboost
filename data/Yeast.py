import scipy
from scipy.io import arff
import pandas as pd
import numpy as np
import os.path as osp


def parse(data):
  class_num = 14

  ori_x = data[:-class_num]
  ori_y = [int(i) * 2 - 1 for i in data[-class_num:]]
  return ori_x, ori_y


def Yeast(data_dir):
  train_file = osp.join(data_dir, 'yeast-train.arff')
  test_file = osp.join(data_dir, 'yeast-test.arff')

  train_data, _ = scipy.io.arff.loadarff(train_file)
  train_data = pd.DataFrame(data=train_data).values
  x_train, y_train = [], []
  for data in train_data:
    x_array, y_array = parse(data)
    x_train.append(x_array)
    y_train.append(y_array)
  x_train = np.array(x_train)
  y_train = np.array(y_train, dtype=np.int8)

  test_data, _ = scipy.io.arff.loadarff(test_file)
  test_data = pd.DataFrame(data=test_data).values
  x_test, y_test = [], []
  for data in test_data:
    x_array, y_array = parse(data)
    x_test.append(x_array)
    y_test.append(y_array)
  x_test = np.array(x_test)
  y_test = np.array(y_test, dtype=np.int8)

  return x_train, x_test, y_train, y_test


if __name__ == '__main__':
  x_train, x_test, y_train, y_test = Yeast('/Users/DavidGuan/Desktop/机器学习/homework2/data/yeast/')
  print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
  print(x_train[0])
  print(y_train[0])
  print(x_test[0])
  print(y_test[0])
