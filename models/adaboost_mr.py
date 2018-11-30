'''
Adaboost MR
by Jingyue Gao
2018.11.30
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
  meta_info_train = []

  train_set_len = len(x_train)
  for i in range(train_set_len):
    pos  = []
    neg  = []
    for j in range(class_num):
      x_train_new.append(np.concatenate((x_train[i], [j])))
      if y_train[i][j] == 1:
        pos.append(j)
      else:
        neg.append(j)
    meta_info_train.append((pos,neg))
  x_train_new = np.array(x_train_new)
  y_train_new = y_train.flatten()

  test_set_len = len(x_test)
  for i in range(test_set_len):
    for j in range(class_num):
      x_test_new.append(np.concatenate((x_test[i], [j])))
  x_test_new = np.array(x_test_new)
  y_test_new = y_test.flatten()

  return x_train_new, x_test_new, y_train_new, y_test_new, meta_info_train


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


def adaboost_mr(args):
  epoch_num = args.epoch_num
  class_num = 14

  x_train, x_test, y_train, y_test, meta_info_train = preprocess(args.data_dir, class_num)
  ini_train_num = len(meta_info_train)
  alpha = np.zeros(epoch_num)
  clfs = []
  D = [] # save the weight for each sample i + label j
  pairD = {} # save the weight for each sample i + pos label j + neg label k
  sampleWeight = 1.0/float(ini_train_num)
  for i in range(ini_train_num):
    tmpD = [0]*class_num
    posWeight = sampleWeight/(2*len(meta_info_train[i][0]))
    negWeight = sampleWeight/(2*len(meta_info_train[i][1]))
    pairNum = len(meta_info_train[i][0])*len(meta_info_train[i][1])
    for j in meta_info_train[i][0]:
      tmpD[j] = posWeight
    for j in meta_info_train[i][1]:
      tmpD[j] = negWeight
    for j in meta_info_train[i][0]:
      for k in meta_info_train[i][1]:
        key = '{}_{}_{}'.format(i,j,k)
        pairD[key] = sampleWeight/pairNum
    D.extend(tmpD)
  result = []
  for i in range(epoch_num):
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(x_train, y_train, sample_weight=D)
    predicted = clf.predict(x_train)
    r = 0.0
    for s in range(ini_train_num):
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          r += pairD[key]*(predicted[s*class_num+j]-predicted[s*class_num+k])
    r = r/2.0
    a = 0.5 * np.log((1 + r) / (1 - r))
    alpha[i] = a
    clfs.append(clf)
    sum_pairD = 0
    for s in range(ini_train_num):
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          pairD[key] = pairD[key]*np.exp(0.5*a*(predicted[s*class_num+k]-predicted[s*class_num+j]))
          sum_pairD += pairD[key]
    D = []
    for s in range(ini_train_num):
      tmpD = [0]*class_num
      for j in meta_info_train[s][0]:
        for k in meta_info_train[s][1]:
          key = '{}_{}_{}'.format(s,j,k)
          pairD[key] = pairD[key]/sum_pairD
          tmpD[j] += pairD[key]/2.0
          tmpD[k] += pairD[key]/2.0
      D.extend(tmpD)
    if args.VIS:
      predicted, accuracy = postprocess(x_test, y_test, clfs, alpha)
      result.append([i + 1, accuracy])
      print('{} {}'.format(i+1,accuracy))
  if args.VIS:
    draw_line_chart(DataFrame(result, columns=['the number of epochs', 'accuracy']), 'the number of epochs', 'accuracy', 'Adaboost.MR')
  predicted, accuracy = postprocess(x_test, y_test, clfs, alpha)
  save_results(predicted, osp.join(args.results_dir, 'AdaboostMR.txt'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch_num', type=int, default=10)
  parser.add_argument('--data_dir', type=str, default='/Users/gaojingyue/Desktop/adaboost/data/yeast/')
  parser.add_argument('--results_dir', type=str, default='/Users/gaojingyue/Desktop/adaboost/results/')
  parser.add_argument('--VIS', action='store_true')
  adaboost_mr(parser.parse_args())