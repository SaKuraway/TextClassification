# coding: UTF-8
import time
import torch
import numpy as np
from train_eval_predict import train, predict, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


# from bert_RNN import Model

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, default='bert_RNN' ,help='choose a model: Bert, ERNIE')
# args = parser.parse_args()
class IntentCLF():
    def __init__(self, device):
        dataset = 'data'  # 数据集
        model_name = 'bert'  # args.model
        x = import_module('models.' + model_name)
        # 读取参数
        self.device = device
        self.config = x.Config(dataset, self.device)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        # 装载model
        self.model = x.Model(self.config).to(self.config.device)

    def train_dev(self):
        start_time = time.time()
        print("Loading data...")
        train_data, dev_data, test_data = build_dataset(self.config)
        print('train_data:', train_data[:2])
        print('dev_data:', dev_data[:2])
        print('test_data:', test_data[:2])
        train_iter = build_iterator(train_data, self.config)
        dev_iter = build_iterator(dev_data, self.config)
        test_iter = build_iterator(test_data, self.config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        train(self.config, self.model, train_iter, dev_iter, test_iter)

    def run_predict(self, text):
        # 加载参数
        self.model.load_state_dict(torch.load(self.config.save_path, map_location=self.device))
        self.model.eval()
        # 预测结果
        return predict(self.config, self.model, text, device=self.device)

if __name__ == '__main__':
    gddw_intent = IntentCLF('cuda')
    # Run_train
    gddw_intent.train_dev()
    while True:
        text = input('请输入需要预测意图的句子:')
        gddw_intent.run_predict(text)
