#!/usr/bin/env python
# encoding: utf-8
'''
@file: stack_lr.py
@time: 2021/7/21 15:27
@author: SaKuraPan_
@desc:
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 去停用词
stopwords = []
with open("./lib/stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())


def stackingLR_train(x_train, y_train, output_path='./model_output/'):
    print('x_train, y_train:', x_train, y_train)
    output_model = output_path + 'stacking_lr.model'
    lr_clf = LogisticRegression(C=1.0)
    lr_clf.fit(x_train, y_train)
    # 评估训练集准确率
    from sklearn import metrics
    result = lr_clf.predict(x_train)  # 逻辑回归预测
    joblib.dump(lr_clf, output_model)
    print("stackingLR训练集准确率:", metrics.accuracy_score(y_train, result))
    print(metrics.classification_report(y_train, result))


def stackingLR_dev(x_dev, y_dev, output_path='./model_output/'):
    # 评估训练集准确率
    from sklearn import metrics
    model = output_path + 'stacking_lr.model'
    lr_clf = joblib.load(model)
    result = lr_clf.predict(x_dev)  # 逻辑回归预测
    print("stackingLR训练集准确率:", metrics.accuracy_score(y_dev, result))
    print(metrics.classification_report(y_dev, result))


def stackingLR_eval(x_dev, y_dev, output_path='./model_output/'):
    # 评估测试集准确率
    from sklearn import metrics
    model = output_path + '_stacking_lr.model'
    lr_clf = joblib.load(model)
    result = lr_clf.predict(x_dev)  # 朴素贝叶斯预测
    print("stackingLR训练集准确率:", metrics.accuracy_score(y_dev, result))
    print(metrics.classification_report(y_dev, result))


def stackingLR_predict(x_test, output_path='./model_output/'):
    model = output_path + 'stacking_lr.model'
    lr_clf = joblib.load(model)
    return lr_clf.predict_proba(x_test)


if __name__ == '__main__':
    stackingLR_train()
