#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson63-cnn.py
@time: 2020/7/23 18:54
@project: deeplearning-with-pytorch-notes
@desc: 第63-65课：CNN
"""
import torch
from torch.nn import functional as F
from torch import nn

print("-----nn.Conv2d-----")
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)

out = layer.forward(x)
print("stride=1, padding=0: out.shape =", out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print("stride=1, padding=1: out.shape =", out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print("stride=2, padding=1: out.shape =", out.shape)
print()

print("-----Inner weight & bias-----")
print("layer.weight =\n", layer.weight)
print("layer.weight.shape =", layer.weight.shape)
print("layer.bias.shape =", layer.bias.shape)
print()

print("-----F.conv2d-----")
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
x = torch.randn(1, 3, 28, 28)

out = F.conv2d(x, w, b, stride=1, padding=1)
print("F.conv2d, stride=1, padding=1: out.shape =", out.shape)
out = F.conv2d(x, w, b, stride=2, padding=2)
print("F.conv2d, stride=2, padding=2: out.shape =", out.shape)
print()
