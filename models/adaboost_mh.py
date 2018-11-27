'''
Adaboost MH
'''
from ..data import create
import numpy as np
from sklearn.tree import DecisionTreeClassifier


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


def postprocess(x_test, y_test, clfs, alpha, class_num):
  epoch_num = len(alpha)
  assert len(clfs) == len(alpha)
  predicted = [clfs[i].predict(x_test) * alpha[i] for i in range(epoch_num)]
  predicted = np.array(predicted)
  predicted = np.sum(predicted, axis=0)
  predicted = np.array([1 if i else -1 for i in predicted], dtype=np.int8)

  accuracy = np.sum(predicted == y_test) / len(y_test)

  print('accuracy: {}'.format(accuracy))


def adaboost_mh(data_dir):
  epoch_num = 10
  class_num = 14

  x_train, x_test, y_train, y_test = preprocess(data_dir, class_num)
  train_set_len = len(x_train)
  alpha = np.zeros(epoch_num)
  clfs = []
  D = np.ones(train_set_len) / train_set_len

  for i in range(epoch_num):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train, sample_weight=D)
    predicted = clf.predict(x_train)

    r = np.sum(D[i] * y_train * predicted)
    assert r < 1
    z = np.sqrt(r)
    a = 0.5 * np.log((1 + r) / (1 - r))
    alpha[i] = a
    clfs.append(clf)

    D = D * np.exp(-a * y_train * predicted) / z

  postprocess(x_test, y_test, clfs, alpha, class_num)

if __name__ == '__main__':
  # x_train, x_test, y_train, y_test = preprocess('/Users/DavidGuan/Desktop/机器学习/homework2/data/yeast/', 14)
  # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
  # print(x_train[0])
  # print(x_train[1])
  # print(y_train[0:14])
  # print(x_test[0])
  # print(x_test[1])
  # print(y_test[0:14])
  adaboost_mh('/Users/DavidGuan/Desktop/机器学习/homework2/data/yeast/')