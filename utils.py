# coding: UTF-8
import time
import torch
from tqdm import tqdm
from random import shuffle
from datetime import timedelta
from collections import OrderedDict

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def Shuffle(self, x, y, random=None, int=int):
    """x, random=random.random -> shuffle list x in place; return None.

    Optional arg random is a 0-argument function returning a random
    float in [0.0, 1.0); by default, the standard random.random.
    """

    if random is None:
        random = self.random  # random=random.random
    # 转成numpy
    if torch.is_tensor(x) == True:
        if self.use_cuda == True:
            x = x.cpu().numpy()
        else:
            x = x.numpy()
    if torch.is_tensor(y) == True:
        if self.use_cuda == True:
            y = y.cpu().numpy()
        else:
            y = y.numpy()
    # 开始随机置换
    for i in range(len(x)):
        j = int(random() * (i + 1))
        if j <= len(x) - 1:  # 交换
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]

    # 转回tensor
    if self.use_cuda == True:
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()

    else:
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    return x, y


def build_dataset(config):
    def load_dataset(path, pad_size=50):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin or 'label,text' in lin:
                    continue
                # print(lin)
                label, content = lin.split(',')
                # print('label, content:', label, content)
                token = config.tokenizer.tokenize(content)
                # token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, get_label_id(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 打乱数据
    shuffle(train)
    shuffle(dev)
    shuffle(test)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        print('batches,batch_size,n_batches:', len(batches), batch_size, self.n_batches)
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    return timedelta(seconds=int(round(time_dif)))

def get_label_id(label,class_path='./data/class.txt'):
    """
    :param search_title:  分类label对应ID
    :return:  label_dict
    """
    classify_dict = {}
    class_list = open(class_path,'r',encoding='utf-8').readlines()
    for class_ in class_list:
        class_index, class_name = class_.split(',')
        classify_dict[class_name.strip()] = int(class_index)
    return classify_dict[label]

def get_label(id,class_path='./data/class.txt'):
    """
    :param search_title:  分类label对应
    :return:  label_dict
    """
    classify_dict = OrderedDict()
    class_list = open(class_path,'r',encoding='utf-8').readlines()
    for class_ in class_list:
        class_index, class_name = class_.split(',')
        classify_dict[int(class_index)] = class_name.strip()
    return classify_dict[id]