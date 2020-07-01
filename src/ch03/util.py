#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: util.py
@time: 2020/7/1 13:44
@project: deeplearning-with-pytorch-notes
@desc:
"""

import torch
from matplotlib import pyplot as plt


def plot_curve(data):
    """
    下降曲线的绘制
    :param data:
    :return:
    """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    """
    可视化识别结果
    :param img:
    :param label:
    :param name:
    :return:
    """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    """
    one_hot编码
    :param label:
    :param depth:
    :return:
    """
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
