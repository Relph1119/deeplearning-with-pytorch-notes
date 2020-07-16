#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson41-perceptron.py
@time: 2020/7/15 19:09
@project: deeplearning-with-pytorch-notes
@desc: 第41课：感知机的梯度推导
"""

import torch
from torch.nn import functional as F

print("-----single output perceptron-----")

x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x @ w.t())
print("o.shape =", o.shape)

loss = F.mse_loss(torch.ones(1, 1), o)
print("loss.shape =", loss.shape)

loss.backward()
print("w.grad =\n", w.grad)
print()

print("-----multi output perceptron-----")
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)

o = torch.sigmoid(x @ w.t())
print("o.shape =", o.shape)

loss = F.mse_loss(torch.ones(1, 2), o)
print("loss =", loss)

loss.backward()
print("w.grad =\n", w.grad)
