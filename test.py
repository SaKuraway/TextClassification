#!/usr/bin/env python
# encoding: utf-8
'''
@file: test.py
@time: 2021/6/18 8:51
@author: SaKuraPan_
@desc:
'''
import requests

url = 'http://0.0.0.0:27519/1000/intent'
test_list = ['连接扫描仪','密码过期','密码过期','奖励管理','工作台','帮助','WiFi、无线网络','文件无法打开','邮件撤回','密码过期','4A被篡改','无法登录数据中心','移动应用','IE浏览器','Excel表报错','Excel表报错','Excel表报错','人资系统绩效考核','考勤问题','南网云网络策略']
test_list = ['招聘系统','招聘系统、帮助','合同系统','二级专区订单收货','智瞰','南网智瞰','智慧搜索','数据中心','督查督办','车辆系统','用车申请','用车流程','稿纸','浏览器','往来单位','重装系统','重装系统','重装系统','打印机卡纸','打印机卡纸','打印机卡纸','重置AD域密码','重置短信平台密码','重置4A密码']
for text in test_list:
    data = {
        'text': text,
        'model': 'STACK'
    }
    print(requests.post(url=url, data=data).json()['Intent'])

