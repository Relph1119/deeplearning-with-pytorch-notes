#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson94-LSTM-Layer.py
@time: 2020/7/27 0:26
@project: deeplearning-with-pytorch-notes
@desc: 第94课：LSTM Layer使用
"""
import torch
from torch import nn

print("-----nn.LSTM-----")
lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print("lstm =", lstm)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print("out.shape =", out.shape)
print("h.shape =", h.shape)
print("c.shape =", c.shape)
print()

print("-----one layer lstm-----")
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)

for xt in x:
    h, c = cell(xt, [h, c])
print("h.shape =", h.shape)
print("c.shape =", c.shape)
print()

print("-----two layer lstm-----")
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)

for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])
print("h2.shape =", h2.shape)
print("c2.shape =", c2.shape)
