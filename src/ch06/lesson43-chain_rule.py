#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson43-chain_rule.py
@time: 2020/7/15 19:44
@project: deeplearning-with-pytorch-notes
@desc: 第43课：链式法则
"""

import torch

x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

y1 = x * w1 + b1
y2 = y1 * w2 + b2

dy2_dy1 = torch.autograd.grad(y2, [y1], retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y1, [w1], retain_graph=True)[0]
dy2_dw1 = torch.autograd.grad(y2, [w1], retain_graph=True)[0]

print("dy2_dy1*dy1_dw1=", dy2_dy1 * dy1_dw1)
print("dy2_dw1 =", dy2_dw1)
