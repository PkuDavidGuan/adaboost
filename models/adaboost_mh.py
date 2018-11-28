'''
Adaboost MH
'''
import argparse
import os.path as osp

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from pandas import DataFrame

from ..data import create
from ..utils.utils import draw_line_chart, save_results


def preprocess(data_dir, class_num):
  x_train, x_test, y_train, y_test = create('yeast', data_dir)
  x_train_new, x_test_new, y_train_new, y_test_new = [], [], [], []

  train_set_len = len(x_train)
  for i in range(train_set_len):
    for j in range(class_num):
      x_train_new.append(np.concatenate((x_train[i], [j])))
  x_train_new = np.array(x_train_new)
  y_train_new = y_train.flatten()

  test_set_len = len(x_test)
  for i in range(test_set_len):
    for j in range(class_num):
      x_test_new.append(np.concatenate((x_test[i], [j])))
  x_test_new = np.array(x_test_new)
  y_test_new = y_test.flatten()

  return x_train_new, x_test_new, y_train_new, y_test_new


def postprocess(x_test, y_test, clfs, alpha):
  class_num = 14
  epoch_num = len(clfs)

  predicted = [clfs[i].predict(x_test) * alpha[i] for i in range(epoch_num)]
  predicted = np.array(predicted)
  predicted = np.sum(predicted, axis=0)
  predicted = np.array([1 if i >= 0 else -1 for i in predicted], dtype=np.int8)

  accuracy = np.sum(predicted == y_test) / len(y_test)

  print('Epoch: {}, accuracy: {}'.format(epoch_num, accuracy))
  predicted = predicted.reshape([-1, class_num]).astype(np.int8)
  return predicted, accuracy


def adaboost_mh(args):
  epoch_num = args.epoch_num
  class_num = 14

  x_train, x_test, y_train, y_test = preprocess(args.data_dir, class_num)
  train_set_len = len(x_train)
  alpha = np.zeros(epoch_num)
  clfs = []
  D = np.ones(train_set_len) / train_set_len
  result = []

  for i in range(epoch_num):
    clf = DecisionTreeClassifier(max_depth=20)
    # clf = GaussianNB()
    # clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(x_train, y_train, sample_weight=D)
    predicted = clf.predict(x_train)

    r = np.sum(D * y_train * predicted)

    # z = np.sqrt(1 - r**2)
    a = 0.5 * np.log((1 + r) / (1 - r))
    alpha[i] = a
    clfs.append(clf)

    D = D * np.exp(-a * y_train * predicted)
    sum = np.sum(D)
    D = D / sum

    if args.VIS:
      predicted, accuracy = postprocess(x_test, y_test, clfs, alpha)
      result.append([i + 1, accuracy])
  if args.VIS:
    draw_line_chart(DataFrame(result, columns=['the number of epochs', 'accuracy']), 'the number of epochs', 'accuracy', 'Adaboost.MH')
  predicted, accuracy = postprocess(x_test, y_test, clfs, alpha)
  save_results(predicted, osp.join(args.results_dir, 'AdaboostMH.txt'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch_num', type=int, default=10)
  parser.add_argument('--data_dir', type=str, default='/Users/DavidGuan/Desktop/机器学习/homework2/data/yeast/')
  parser.add_argument('--results_dir', type=str, default='/Users/DavidGuan/Desktop/机器学习/homework2/results/')
  parser.add_argument('--VIS', action='store_true')
  adaboost_mh(parser.parse_args())