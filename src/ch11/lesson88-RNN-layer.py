#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson88-RNN-layer.py
@time: 2020/7/24 14:33
@project: deeplearning-with-pytorch-notes
@desc: 第88-89课：RNN Layer使用
"""
import torch
from torch import nn

print("-----RNN------")
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x, torch.zeros(1, 3, 20))
print("out.shape =", out.shape)
print("h.shape =", h.shape)
print()

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x, torch.zeros(4, 3, 20))
print("out.shape =", out.shape)
print("h.shape =", h.shape)
print()
# print(vars(rnn))

print('-----Rnn by Cell-----')

cell1 = nn.RNNCell(100, 20)
h1 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
print("h1.shape =", h1.shape)

cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
print("h1.shape =", h2.shape)
print()

print('-----LSTM-----')
lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print("out.shape =", out.shape)
print("h.shape =", h.shape)
print("c.shape =", c.shape)
print()

print('-----One Layer LSTM-----')
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])
print("h.shape =", h.shape)
print("c.shape =", c.shape)
print()

print('-----Two Layer LSTM-----')
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
