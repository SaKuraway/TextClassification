# Bert-Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

文件说明：
├── Bert_LSTM_Clf_Pytorch
│   ├── app_service.py: 1000号意图识别的flask服务，提供接口，接口参数有model和text（详见接口文档）。将预测的结果对应为中文label及概率proba返回给用户。
│   ├── bert_pretrain: BERT与训练模型的文件
│   ├── config.ini: 配置文件
│   ├── data: 训练、验证及预测的数据
│   ├── dnn_main.py: 深度学习dnn（Bert+Fc）的训练、验证及预测的调用代码
│   ├── document: 原始项目相关文档
│   ├── intent_fasttext.py: 使用fasttext的数据处理、训练及预测调用代码(本次上线没用到)
│   ├── lib: 相关资源库，如停用词典、分词词典等
│   ├── model_output: 模型输出目录文件夹
│   ├── models: bert相关深度学习的代码
│   ├── pytorch_pretrained: pytorch的预训练代码
│   ├── README.md: 本说明文档
│   ├── run_main.py: 启动app_server的代码
│   ├── run_main.sh: 启动run_main的bash代码
│   ├── stacking_lr.py: 使用LR作为集成模型stacking的次级学习器，提供LR的训练、验证及预测的调用代码
│   ├── stacking_main.py: 使用bert、svm、rfc作为初级学习器，其softmax预测结果输入到次级学习器LR中，提供stacking的训练、验证及预测的调用代码
│   ├── test.py: 部署后的http调用测试代码
│   ├── tfidf_nb.py: 使用tfidf作为特征，朴素贝叶斯NaiveBayes作为学习器，提供其训练、验证及预测的调用代码(本次上线没用到)
│   ├── tfidf_rfc.py: 使用tfidf作为特征，随机森林RandomForest作为学习器，提供其训练、验证及预测的调用代码
│   ├── tfidf_svm.py: 使用tfidf作为特征，支持向量机SVM作为学习器，提供其训练、验证及预测的调用代码
│   ├── train_eval_predict.py: （Bert+Fc）的训练、验证及预测的实际执行调用代码
│   ├── utils.py: （Bert+Fc）的数据预处理代码
│   └── version.txt: 版本说明
└── docker_run.cmd: docker容器的运行命令