# coding=utf-8
from flask import Flask, render_template, request, jsonify
import configparser
from tfidf_nb import model_predict as nb_predict
from tfidf_rfc import model_predict as rfc_predict
from dnn_main import IntentCLF
from stacking_main import stacking_predict
from multiprocessing import Pool

# 读取配置文件
conf = configparser.ConfigParser()
conf.read("config.ini")
is_debug = conf.get("Basic", "debug")
class_path = conf.get("Basic", "class_path")
pool_num = conf.getint('Basic', 'pool')
pool = Pool(processes=pool_num)
if is_debug == 'True':
    host = conf.get("Server_dev", "host")
    port = conf.get("Server_dev", "port")
else:
    host = conf.get("Server_pro", "host")
    port = conf.get("Server_pro", "port")
print(is_debug, class_path, host, port)

# 初始化函数
app = Flask(__name__, static_url_path="/static")
app.config['JSON_AS_ASCII'] = False
gddw_intent = IntentCLF('cpu')


def get_classify_dict(class_path):
    """
    :param search_title:  分类label对应classification
    :return:  label_dict
    """
    return {int(label_class.split(',')[0]): label_class.split(',')[-1].strip() for label_class in open(class_path, 'r', encoding='utf-8').readlines()}


classify_dict = get_classify_dict(class_path)


@app.route('/1000/intent', methods=['POST'])
def Gddw_intent(code=1, message=None, intent=None, proba=None):
    """
    前端调用接口。
    路径：/gddw/intent
    请求方式：POST
    请求参数：text,model
    :return: Best similar intent prediction and its probability.
    """
    # 判断从请求中获取参数信息是否为空
    model = request.form['model'] if 'model' in list(request.form.keys()) else 'STACK'  # RFC
    try:
        text = request.form['text'].strip()
        if model == 'STACK':
            prediction, label_proba = stacking_predict(sentence_list=[text], pool=pool)
            for y in prediction:
                predict_proba = {classify_dict[i]: label_prob_list for i, label_prob_list in enumerate(y)}
                result = sorted(predict_proba.items(), key=lambda x: x[1], reverse=True)[0]
                message = 'Successed.'
                intent = result[0]
                proba = str(result[1])
        elif model == 'BERT':
            # Bert + BiLSTM
            prediction = gddw_intent.run_predict(text)
            for y in prediction:
                predict_proba = {classify_dict[i]: label_prob_list for i, label_prob_list in enumerate(y)}
                result = sorted(predict_proba.items(), key=lambda x: x[1], reverse=True)[0]
                message = 'Successed.'
                intent = result[0]
                proba = str(result[1])
        elif model == 'NB':
            # Tfidf + NaiveBayes
            model = 'Tfidf + NaiveBayes'
            prediction = nb_predict([text])
            for y in prediction:
                predict_proba = {classify_dict[i]: label_prob_list for i, label_prob_list in enumerate(y)}
                result = sorted(predict_proba.items(), key=lambda x: x[1], reverse=True)[0]
                message = 'Successed.'
                intent = result[0]
                proba = str(result[1])
        elif model == 'SVM':
            # Tfidf + NaiveBayes
            model = 'Tfidf + SVM'
            prediction = nb_predict([text])
            for y in prediction:
                predict_proba = {classify_dict[i]: label_prob_list for i, label_prob_list in enumerate(y)}
                result = sorted(predict_proba.items(), key=lambda x: x[1], reverse=True)[0]
                message = 'Successed.'
                intent = result[0]
                proba = str(result[1])
        elif model == 'RFC':
            # Tfidf + RandomRorestClassifier
            model = 'Tfidf + RandomRorest'
            prediction = rfc_predict([text])
            for y in prediction:
                predict_proba = {classify_dict[i]: label_prob_list for i, label_prob_list in enumerate(y)}
                result = sorted(predict_proba.items(), key=lambda x: x[1], reverse=True)[0]
                message = 'Successed.'
                intent = result[0]
                proba = str(result[1])
        else:
            # No-model Selected.
            code = 0
            message = 'Model Selected Error! ' + str(model)
    except Exception as e:
        # exception
        code = 0
        message = 'Requests Failed! ' + str(e)
    else:
        print(result)
    finally:
        # return final result.
        return jsonify({'code': code, 'message': message + ' With using [{0}] model.'.format(model), 'Intent': intent,
                        'Proba': proba})


@app.route("/")
def index():
    return render_template("/index.html")


def runmain():
    # 启动app服务
    app.run(host=host, port=int(port))


# 启动APP
if __name__ == "__main__":
    runmain()
