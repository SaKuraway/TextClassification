#!/usr/bin/env python
# encoding: utf-8
'''
@file: stacking_lr.py
@time: 2021/7/19 17:24
@author: SaKuraPan_
@desc:
'''
import pandas as pd
import jieba, time
from numpy import array
from numpy import concatenate as np_concatenate
from utils import get_label
from dnn_main import IntentCLF
from tfidf_rfc import model_predict as rfc_predict
from tfidf_nb import model_predict as nb_predict
from tfidf_svm import model_predict as svm_predict
from stacking_lr import stackingLR_train, stackingLR_dev, stackingLR_predict
from multiprocessing import Pool
from configparser import ConfigParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib, pickle

# 读取配置文件
cfg = ConfigParser()
cfg.read('config.ini')
pool_num = cfg.getint('Basic', 'pool')
pd.set_option('mode.chained_assignment', None)
jieba.load_userdict('./lib/user_dict.txt')
# 预加载模型（提高predict效率）
vectorizer = CountVectorizer(decode_error="replace",
                             vocabulary=pickle.load(open('./model_output/vectorizer.pkl', "rb+")))
tfidf = joblib.load('./model_output/tfidf.pkl')
svm_clf = joblib.load('./model_output/svm_cls.pkl')
rfc_clf = joblib.load('./model_output/rfc_cls.pkl')
nb_clf = joblib.load('./model_output/nb_cls.pkl')
bert_model = IntentCLF('cpu')

# 去停用词
stopwords = []
with open("./lib/stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())


def get_data(data_path='./data/'):
    # 读取csv数据
    train_df = pd.read_csv(data_path + 'train.csv').astype(str)
    dev_df = pd.read_csv(data_path + 'dev.csv').astype(str)
    test_df = pd.read_csv(data_path + 'test.csv').astype(str)
    # 拼接
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_dev = dev_df['text'].tolist()
    y_dev = dev_df['label'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    return (X_train, y_train, X_dev, y_dev, X_test, y_test)


def bert_fc(sentence_list):
    # albert + fc
    t0 = time.time()
    # bert_model = IntentCLF('cpu')
    print('bert模型加载耗时', time.time() - t0)
    proba_bert = array([bert_model.run_predict(sentence)[0] for sentence in sentence_list])
    print('bert shape:', proba_bert.shape)
    print('bert耗时', time.time() - t0)
    return proba_bert


def w2v_textcnn(sentence_list, model):
    # w2v_textcnn
    cnn_model = IntentCLF('cpu', model)
    proba_textcnn = array([cnn_model.predict(sentence)[0] for sentence in sentence_list])
    print(model, 'textcnn shape:', proba_textcnn.shape)
    return proba_textcnn


def tfidf_svm(sentence_list):
    t0 = time.time()
    # proba_svm = array(svm_predict(sentence_list))
    sentence_list = [' '.join(
        [words.strip() for words in list(jieba.lcut(sents)) if words not in stopwords]) for
                     sents in sentence_list]
    vector = vectorizer.transform(sentence_list)
    sentence_tfidf = tfidf.transform(vector)
    proba_svm = array(svm_clf.predict_proba(sentence_tfidf))  # SVM预测
    print('svm_proba shape:', proba_svm.shape)
    print('svm耗时', time.time() - t0)
    return proba_svm


def tfidf_rfc(sentence_list):
    t0 = time.time()
    # proba_rfc = array(rfc_predict(sentence_list))
    sentence_list = [' '.join(
        [words.strip() for words in list(jieba.lcut(sents)) if words not in stopwords]) for
                     sents in sentence_list]
    vector = vectorizer.transform(sentence_list)
    sentence_tfidf = tfidf.transform(vector)
    proba_rfc = array(rfc_clf.predict_proba(sentence_tfidf))  # 随机森林预测
    print('rfc_proba shape:', proba_rfc.shape)
    print('rfc耗时', time.time() - t0)
    return proba_rfc


def tfidf_nb(sentence_list):
    # proba_nb = array(nb_predict(sentence_list))
    sentence_list =[' '.join([words.strip() for words in list(jieba.lcut(sents)) if words not in stopwords]) for sents in sentence_list]
    vector = vectorizer.transform(sentence_list)
    sentence_tfidf = tfidf.transform(vector)
    proba_nb = nb_clf.predict_proba(sentence_tfidf)  # 朴素贝叶斯预测
    print('nb_proba shape:', proba_nb.shape)
    return proba_nb


def stacking_train(X_train, y_train):
    bert_result = bert_fc(X_train)
    tfidf_svm_result = tfidf_svm(X_train)
    tfidf_rfc_result = tfidf_rfc(X_train)
    # tfidf_nb_result = tfidf_nb(X_train)
    # textcnn_title_result = w2v_textcnn(X_title, 'w2v_TextCNN_{0}_title'.format(resource))  # 默认title
    x_train = np_concatenate((bert_result, tfidf_svm_result, tfidf_rfc_result), axis=1)
    print('x_train:', x_train)
    return stackingLR_train(x_train, y_train)


def stacking_dev(X_dev, y_dev):
    bert_result = bert_fc(X_dev)
    tfidf_svm_result = tfidf_svm(X_dev)
    tfidf_rfc_result = tfidf_rfc(X_dev)
    # tfidf_nb_result = tfidf_nb(X_dev)
    # textcnn_title_result = w2v_textcnn(X_title, 'w2v_TextCNN_{0}_title'.format(resource))  # 默认title
    x_dev = np_concatenate((bert_result, tfidf_svm_result, tfidf_rfc_result), axis=1)
    print('x_dev:', x_dev)
    return stackingLR_dev(x_dev, y_dev)


def model_call(sentence_list, model_name):
    if model_name == 'bert_fc':
        return bert_fc(sentence_list)
    elif model_name == 'w2v_TextCNN':
        return w2v_textcnn(sentence_list, 'w2v_TextCNN')
    elif model_name == 'tfidf_svm':
        return tfidf_svm(sentence_list)
    elif model_name == 'tfidf_nb':
        return tfidf_nb(sentence_list)
    elif model_name == 'tfidf_rfc':
        return tfidf_rfc(sentence_list)
    else:
        return


def stacking_predict(sentence_list, top_k=None, pool=None):
    t0 = time.time()
    if pool is None:
        # get model output.
        bert_result = bert_fc(sentence_list)
        tfidf_svm_result = tfidf_svm(sentence_list)
        tfidf_rfc_result = tfidf_rfc(sentence_list)
        # tfidf_nb_result = tfidf_nb(sentence_list)
        # textcnn_title_result = w2v_textcnn(X_title, 'w2v_TextCNN_{0}_title'.format(resource))  # 默认title
    else:
        print('使用多进程..')
        bert_result = bert_fc(sentence_list)  # bert无法多进程加载，不知为啥
        model_params = [(sentence_list, 'tfidf_svm'), (sentence_list, 'tfidf_rfc')]
        tfidf_svm_result, tfidf_rfc_result = pool.starmap(model_call, model_params)  # Proba_list
    output_list = [bert_result, tfidf_svm_result, tfidf_rfc_result]
    # for i in output_list: print('shape:', i.shape)
    x_test = np_concatenate(output_list, axis=1)
    print(len(x_test), 'x_test:', x_test)
    result_list = stackingLR_predict(x_test)
    # 输出导则结果
    top_k_idx_list = [result.argsort()[::-1][0:top_k] for result in result_list]  # 最大概率的前N个result
    label_proba = [{get_label(idx): proba_list[idx] for idx in idx_list} for idx_list, proba_list in
                   zip(top_k_idx_list, result_list)]
    # print('predict result:', label_proba)
    print('Stacking Predict耗时：', time.time() - t0)
    return result_list, label_proba


if __name__ == '__main__':
    # 获取训练数据
    X_train,y_train,X_dev,y_dev,X_test,y_test = get_data()
    # 训练StackingLR模型
    stacking_train(X_train, y_train)
    # 评估模型
    stacking_dev(X_dev, y_dev)
    # exit()
    # 使用Multiprocessing创建多进程池
    pool = Pool(processes=pool_num)
    # 预测StackingLR模型
    result_list, label_proba = stacking_predict(sentence_list=['移动应用', '南网云', '我的电脑登录不了', '我的电脑连不了网', '我的手机连不了网'],
                                                top_k=3, pool=pool)  # ['台风','发改委']
    print('predict result:', label_proba)
