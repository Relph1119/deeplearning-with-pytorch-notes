#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson31-attribute_statistics.py
@time: 2020/7/14 18:44
@project: deeplearning-with-pytorch-notes
@desc: 第31-32课：属性统计
"""

import torch

print("----- norm-p -----")
a = torch.full([8], 1, dtype=torch.float)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print("b =\n", b)
print("c =\n", c)
print("a.norm(1) =", a.norm(1))
print("b.norm(1) =", b.norm(1))
print("c.norm(1) =", c.norm(1))

print("a.norm(2) =", a.norm(2))
print("b.norm(2) =", b.norm(2))
print("c.norm(2) =", c.norm(2))

print("b.norm(1, dim=1) =", b.norm(1, dim=1))
print("b.norm(2, dim=1) =", b.norm(2, dim=1))
print("c.norm(1, dim=0) =\n", c.norm(1, dim=0))
print("c.norm(2, dim=0) =\n", c.norm(2, dim=0))
print()

print("-----mean / sum / min / max / prod-----")
a = torch.arange(8).view(2, 4).float()
print("a =", a)
print("a.min() =", a.min())
print("a.max() =", a.max())
print("a.mean() =", a.mean())
print("a.prod() =", a.prod())
print("a.sum() =", a.sum())
print()

print("-----argmax / argmin-----")
# argmax(), argmin()，首先打平，然后返回的是索引
print("a.argmax() =", a.argmax())
print("a.argmin() =", a.argmin())
a = torch.randn(4, 10)
print("a[0] =\n", a[0])
print("a.argmax() =", a.argmax())
print("a.argmin(dim=1) =", a.argmin(dim=1))
print()

print("-----dim / keepdim-----")
print("a =\n", a)
print("a.max(dim=1) =\n", a.max(dim=1))
print("a.argmax(dim=1) =", a.argmax(dim=1))
print("a.max(dim=1, keepdim=True) =\n", a.max(dim=1, keepdim=True))
print("a.argmax(dim=1, keepdim=True) =\n", a.argmax(dim=1, keepdim=True))

print("-----topk / kthvalue-----")
print("a.topk(3, dim=1) =\n", a.topk(3, dim=1))
print("a.topk(3, dim=1, largest=False) =\n", a.topk(3, dim=1, largest=False))
print("a.kthvalue(8, dim=1) =\n", a.kthvalue(8, dim=1))
print("a.kthvalue(3) =\n", a.kthvalue(3))
print("a.kthvalue(3, dim=1) =\n", a.kthvalue(3, dim=1))
print()

print("----- > / >= / < / <= / != / == -----")
print("a > 0 :\n", a > 0)
print("torch.gt(a, 0) =\n", torch.gt(a, 0))
print("a != 0 :\n", a != 0)
a = torch.ones(2, 3)
b = torch.randn(2, 3)
print("torch.eq(a, b) =\n", torch.eq(a, b))
print("torch.eq(a, a) =\n", torch.eq(a, a))
print("torch.equal(a, a) =", torch.equal(a, a))
