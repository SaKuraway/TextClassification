# from weibo.utils import tokenize, load_curpus
import numpy as np

import pandas as pd
import re
import jieba
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pickle

pd.set_option('mode.chained_assignment', None)
jieba.load_userdict('./lib/user_dict.txt')


def tokenize(text):
    """
    带有语料清洗功能的分词函数
    """
    text = re.sub("\{%.+?%\}", " ", text)  # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)  # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)  # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    icons = re.findall("\[.+?\]", text)  # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)  # 将文本中的图标替换为`IconMark`

    tokens = []
    for k, w in enumerate(jieba.lcut(text)):
        w = w.strip()
        if "IconMark" in w:  # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha():  # 只保留有效文本
            tokens.append(w)
    return tokens


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
    X_train = [' '.join(
        [words.strip() for words in list(jieba.lcut(sents)) if (words not in stopwords) and (str(words) != 'nan')]) for
               sents in train_df['text'].tolist()]
    y_train = train_df['label'].tolist()
    X_dev = [' '.join(
        [words.strip() for words in list(jieba.lcut(sents)) if (words not in stopwords) and (str(words) != 'nan')]) for
             sents in dev_df['text'].tolist()]
    y_dev = dev_df['label'].tolist()
    X_test = [' '.join(
        [words.strip() for words in list(jieba.lcut(sents)) if (words not in stopwords) and (str(words) != 'nan')]) for
              sents in test_df['text'].tolist()]
    y_test = test_df['label'].tolist()
    return (X_train, y_train, X_dev, y_dev, X_test, y_test)


def train_tfidf_nb(data_dir='./data/', model_dir='./model_output/'):
    """
    通过词频-idf的特征输入，训练朴素贝叶斯模型的先验概率。
    :param search_title:  搜索文档的类别
    :return: vectorizer,transformer,word_bags,nb_clf,label_dict
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = get_data(data_dir)

    vectorizer = CountVectorizer(token_pattern='\[?\w+\]?')  #, stop_words=stopwords
    count_vector = vectorizer.fit_transform(X_train)
    # joblib.dump(count_vector, 'vectorizer.m')
    vectorizer_path = model_dir + 'vectorizer.pkl'
    with open(vectorizer_path, 'wb') as fw:
        pickle.dump(vectorizer.vocabulary_, fw)

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_vector)
    # joblib.dump(tfidf,'tfidf.m')
    tfidftransformer_path = model_dir + 'tfidf.pkl'
    with open(tfidftransformer_path, 'wb') as fw:
        pickle.dump(transformer, fw)
    word_bags = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    print('word_bags',word_bags)
    print(tfidf.shape,tfidf)    # (3056, 798),
    from sklearn.naive_bayes import MultinomialNB

    nb_clf = MultinomialNB()
    nb_clf.fit(tfidf, y_train)
    # 评估测试集准确率
    from sklearn import metrics
    vector_cut_words = vectorizer.transform(X_test)
    tfidf_words = transformer.fit_transform(vector_cut_words)  # tf-idf
    result = nb_clf.predict(tfidf_words)  # 朴素贝叶斯预测
    # tfidf_words = transformer.fit_transform(vector_cut_words)   # tf-idf
    print(metrics.classification_report(y_test, result))
    print("测试模型准确率:", metrics.accuracy_score(y_test, result))
    # 保存模型
    nb_clf_path = model_dir + 'nb_cls.pkl'
    with open(nb_clf_path, 'wb') as fw:
        pickle.dump(nb_clf, fw)
    return (vectorizer, transformer, word_bags, nb_clf)


def model_predict(sentence_list):
    vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./model_output/vectorizer.pkl',"rb+")))
    sentence_list =[' '.join([words.strip() for words in list(jieba.lcut(sents)) if (words not in stopwords) and (str(words) != 'nan')]) for sents in sentence_list]
    vectorizer = vectorizer.transform(sentence_list)
    tfidf = joblib.load('./model_output/tfidf.pkl')
    sentence_tfidf = tfidf.transform(vectorizer)
    nb_clf = joblib.load('./model_output/nb_cls.pkl')
    result = nb_clf.predict_proba(sentence_tfidf)  # 朴素贝叶斯预测
    return result


if __name__ == '__main__':
    # 所有模型遍历，查看准确率
    train_tfidf_nb()
    print(model_predict(['移动应用','南网云','点解呢']))