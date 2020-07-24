#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson66-pooling-sample.py
@time: 2020/7/23 19:10
@project: deeplearning-with-pytorch-notes
@desc: 第66课：池化层和采样
"""
import torch
from torch.nn import functional as F
from torch import nn

print("-----Pooling-----")
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=2, padding=2)
x = out
print("x.shape =", x.shape)
layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print("nn.MaxPool2d, stride=2: out.shape =", out.shape)
out = F.avg_pool2d(x, 2, stride=2)
print("F.avg_pool2d, stride=2: out.shape =", out.shape)
print()

print("-----F.interpolate-----")
x = out
out = F.interpolate(x, scale_factor=2, mode='nearest')
print("F.interpolate, scale_factor=2: out.shape =", out.shape)
out = F.interpolate(x, scale_factor=3, mode='nearest')
print("F.interpolate, scale_factor=3: out.shape =", out.shape)
print()

print("-----ReLU-----")
print("x.shape =", x.shape)
layer = nn.ReLU(inplace=True)
out = layer(x)
print("nn.ReLU: out.shape =", out.shape)
out = F.relu(x)
print("F.relu: out.shape =", out.shape)

