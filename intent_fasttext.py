#!/usr/bin/env python
# encoding: utf-8
'''
@file: intent_fasttext.py
@time: 2020/10/16 10:20
@author: SaKuraPan_
@desc:
'''
import fasttext
from os import listdir
from jieba import lcut


def data_process(file_type='test'):
    import jieba
    from collections import OrderedDict
    def get_label(id):
        """
        :param search_title:  分类label对应
        :return:  label_dict
        """
        classify_dict = OrderedDict()
        class_list = open('./data/class.txt', 'r', encoding='utf-8').readlines()
        for class_ in class_list:
            class_index, class_name = class_.split(' ')
            classify_dict[int(class_index)] = class_name.strip()
        print(classify_dict)
        return classify_dict[id]
    # processing
    file_list = open('./data/{0}.csv'.format(file_type), 'r', encoding='utf-8').read().split('\n')
    with open('./data/{0}.txt'.format(file_type), 'w', encoding='utf-8') as f:
        for file in file_list:
            if not file or 'label' in file:
                continue
            try:
                text = ' '.join(jieba.lcut(file.split(',')[1]))
                f.write(text + '\t' + '__label__' + get_label(int(file.split(',')[0])) + '\n')
            except Exception as e:
                print(e)
        f.close()


class FasttextIntent(object):
    def __init__(self, model_name='FastTextModel.bin', model_path='./model_output/', data_path='./data/'):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        if model_name in listdir(model_path):
            self.classifier = fasttext.load_model(self.model_path + self.model_name)

    def train(self, file_name='train.txt'):
        self.classifier = fasttext.train_supervised(
            input=self.data_path + file_name,
            label_prefix='__label__',
            dim=256,
            epoch=200,
            lr=0.05,
            lr_update_rate=50,
            min_count=1,
            loss='ns',
            word_ngrams=3,
            bucket=20000)
        self.classifier.save_model(self.model_path + self.model_name)

    def eval(self, file_name='test.txt'):
        testDataFile = self.data_path + file_name
        result = self.classifier.test(testDataFile)
        print('测试集的数据量', result[0])
        print('测试集的准确率', result[1])
        print('测试集的召回率', result[2])
        return result

    def predict(self, input_text):
        result = self.classifier.predict(" ".join(lcut(input_text.replace('\n', ''))))
        return result


if __name__ == '__main__':
    # Data_processing
    data_process(file_type='train')
    # Run Train
    fast_intent = FasttextIntent()
    fast_intent.train()
    # Run Eval
    fast_intent.eval()
    # Run Predict
    predict_result = fast_intent.predict("南网云")
    print('预测标签为:', predict_result[0][0].replace('__label__', ''), ',对应概率为:', predict_result[1][0])
