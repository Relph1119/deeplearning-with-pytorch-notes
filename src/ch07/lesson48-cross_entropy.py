#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson48-cross_entropy.py
@time: 2020/7/17 9:04
@project: deeplearning-with-pytorch-notes
@desc: 第48-49课：交叉熵
"""
import torch
from torch.nn import functional as F

print("-----Lottery-----")
a = torch.full([4], 1 / 4., dtype=torch.float)
print("a =", a)
print("a * torch.log2(a) =", a * torch.log2(a))
print("-(a*torch.log2(a)).sum() =", -(a * torch.log2(a)).sum())
a = torch.tensor([0.1, 0.1, 0.1, 0.7])
print("a =", a)
print("-(a*torch.log2(a)).sum() =", -(a * torch.log2(a)).sum())

print("-----Numerical Stability-----")
x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()
print("logits.shape =", logits.shape)

pred = F.softmax(logits, dim=1)
print("pred.shape =", pred.shape)

pred_log = torch.log(pred)
print("F.cross_entropy(logits, torch.tensor([3])) =", F.cross_entropy(logits, torch.tensor([3])))
print("F.nll_loss(pred_log, torch.tensor([3])) =", F.nll_loss(pred_log, torch.tensor([3])))
