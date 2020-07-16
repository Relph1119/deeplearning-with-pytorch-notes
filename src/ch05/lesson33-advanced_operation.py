#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson33-advanced_operation.py
@time: 2020/7/14 20:03
@project: deeplearning-with-pytorch-notes
@desc: 第33课-高阶操作
"""

import torch

print("-----where-----")
cond = torch.rand(2, 2)
print("cond =\n", cond)
a = torch.zeros(2, 2)
print("a =\n", a)
b = torch.ones(2, 2)
print("b =\n", b)
print("torch.where(cond > 0.5, a, b) =\n", torch.where(cond > 0.5, a, b))
print()

print("-----gather-----")
prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)
print("idx =\n", idx)
idx = idx[1]
print("idx =\n", idx)
label = torch.arange(10) + 100
print("label =\n", label)
print("torch.gather(label.expand(4, 10), dim=1, index=idx.long()) =\n",
      torch.gather(label.expand(4, 10), dim=1, index=idx.long()))
