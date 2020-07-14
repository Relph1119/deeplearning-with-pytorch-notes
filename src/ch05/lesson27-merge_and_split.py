#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson27-merge_and_split.py
@time: 2020/7/14 13:37
@project: deeplearning-with-pytorch-notes
@desc: 第27-28课：合并与分割
"""
import torch

print("-----cat-----")
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
print("a.shape =", a.shape)
print("b.shape =", b.shape)
print("torch.cat([a, b], dim=0).shape =", torch.cat([a, b], dim=0).shape)
print()

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
print("a1.shape =", a1.shape)
print("a2.shape =", a2.shape)
print("torch.cat([a1, a2], dim=0).shape =", torch.cat([a1, a2], dim=0).shape)
a2 = torch.rand(4, 1, 32, 32)
print("a1.shape =", a1.shape)
print("a2.shape =", a2.shape)
print("torch.cat([a1, a2], dim=1).shape =", torch.cat([a1, a2], dim=1).shape)
a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
print("a1.shape =", a1.shape)
print("a2.shape =", a2.shape)
print("torch.cat([a1, a2], dim=2).shape =", torch.cat([a1, a2], dim=2).shape)
print()

# stack会创建一个新的维度
print("-----stack-----")
print("torch.stack([a1, a2], dim=2).shape =", torch.stack([a1, a2], dim=2).shape)
a = torch.rand(32, 8)
b = torch.rand(32, 8)
print("torch.stack([a, b], dim=0).shape =", torch.stack([a, b], dim=0).shape)

print("-----split-----")
print("a.shape =", a.shape)
print("b.shape =", b.shape)
c = torch.stack([a, b], dim=0)
print("c.shape =", c.shape)
aa, bb = c.split([1,1], dim=0)
print("aa.shape, bb.shape =", (aa.shape, bb.shape))
aa, bb = c.split(1, dim=0)
print("aa.shape, bb.shape =", (aa.shape, bb.shape))