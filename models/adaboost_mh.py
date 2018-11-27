'''
Adaboost MH
'''
from ..data import create
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def adaboost_mh(data_dir):
  x_train, x_test, y_train, y_test = create(data_dir)
  epoch_num = 10
  train_set_len = len(x_train)
  alpha = np.zeros(epoch_num)
  h = []
  D = np.ones([epoch_num, train_set_len]) / train_set_len

  for i in range(epoch_num):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train, sample_weight=D)
    predicted = clf.predict(x_train)

    r = np.sum(D[i] * y_train * predicted)
    z = np.sqrt(r)
    a = 0.5 * np.log((1 + r) / (1 - r))
    alpha[i] = a

    if i != epoch_num - 1:
      D[i + 1] = D[i] * np.exp(-a * y_train * predicted) / z


if __name__ == '__main__':
  adaboost_mh('/Users/DavidGuan/Desktop/机器学习/homework2/data/yeast/')