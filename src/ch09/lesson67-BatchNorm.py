#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson67-BatchNorm.py
@time: 2020/7/24 10:06
@project: deeplearning-with-pytorch-notes
@desc: 第67-68课：BatchNorm
"""

import torch
from torch import nn

print("-----nn.BatchNorm1d------")
x = torch.randn(100, 16) + 0.5
layer = torch.nn.BatchNorm1d(16)

print("layer.running_mean =", layer.running_mean)
print("layer.running_var =", layer.running_var)
layer.eval()
print()
out = layer(x)
print("layer(x) after......")
print("layer.running_mean =", layer.running_mean)
print("layer.running_var =", layer.running_var)
print()
x = torch.randn(100, 16) + 0.5
layer = torch.nn.BatchNorm1d(16)
for i in range(100):
    out = layer(x)

print("loop 100 times, layer(x) after......")
print("layer.running_mean =", layer.running_mean)
print("layer.running_var =", layer.running_var)
print()

print("-----nn.BatchNorm2d------")
x = torch.randn(1, 16, 7, 7)
print("x.shape =", x.shape)
layer = nn.BatchNorm2d(16)
out = layer(x)
print("out.shape =", out.shape)
print("layer.weight =\n", layer.weight)
print("layer.weight.shape =", layer.weight.shape)
print("layer.bias.shape =", layer.bias.shape)
print("vars(layer) =\n", vars(layer))
print()
