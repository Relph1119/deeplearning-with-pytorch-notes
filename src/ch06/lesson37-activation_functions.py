#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson37-activation_functions.py
@time: 2020/7/15 14:01
@project: deeplearning-with-pytorch-notes
@desc: 第37-40课：激活函数和Loss的梯度
"""
import torch
from torch.nn import functional as F

print("-----torch.sigmoid------")
a = torch.linspace(-100, 100, 10)
print("a =\n", a)
print("torch.sigmoid(a) =\n", torch.sigmoid(a))
print()

print("-----torch.tanh-----")
a = torch.linspace(-1, 1, 10)
print("a =\n", a)
print("torch.tanh(a) =\n", torch.tanh(a))
print()

print("-----F.relu-----")
a = torch.linspace(-1, 1, 10)
print("a =\n", a)
print("torch.relu(a) =\n", torch.relu(a))
print("F.relu(a) =\n", F.relu(a))

import torch
from torch.nn import functional as F

print("-----autograd.grad-----")
x = torch.ones(1)
w = torch.full([1], 2, requires_grad=True, dtype=torch.float)
print("x =", x)
print("w =", x)
print("w.requires_grad_() =\n", w.requires_grad_())
mse = F.mse_loss(torch.ones(1), x * w)
print("mse =", mse)
print("torch.autograd.grad(mse, [w]) =", torch.autograd.grad(mse, [w]))
print()

print("-----loss.backword-----")
mse = F.mse_loss(torch.ones(1), x * w)
mse.backward()
print("w.grad =", w.grad)
print()

print("-----F.softmax-----")
a = torch.rand(3, requires_grad=True)
print("a.requires_grad_() =\n", a.requires_grad_())
p = F.softmax(a, dim=0)
print("torch.autograd.grad(p[1], [a], retain_graph=True) =\n", torch.autograd.grad(p[1], [a], retain_graph=True))
print("torch.autograd.grad(p[2], [a]) =\n", torch.autograd.grad(p[2], [a]))
