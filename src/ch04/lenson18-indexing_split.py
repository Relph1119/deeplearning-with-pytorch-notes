#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lenson18-indexing_split.py
@time: 2020/7/12 10:14
@project: deeplearning-with-pytorch-notes
@desc: 第18-19课：索引与切片
"""
import torch

print("-----Indexing-----")
a = torch.rand(4, 3, 28, 28)
print("a[0].shape =", a[0].shape)
print("a[0, 0].shape =", a[0, 0].shape)
print("a[0, 0, 2, 4] =", a[0, 0, 2, 4])
print()

print("-----select first / last N-----")
print("a.shape =", a.shape)
print("a[:2].shape =", a[:2].shape)
print("a[:2, :1, :, :].shape =", a[:2, :1, :, :].shape)
print("a[:2, 1:, :, :].shape =", a[:2, 1:, :, :].shape)
print("a[:2, -1:, :, :].shape =", a[:2, -1:, :, :].shape)
print()

print("-----select by steps-----")
print("a[:, :, 0:28:2, 0:28:2].shape =", a[:, :, 0:28:2, 0:28:2].shape)
print("a[:, :, ::2, ::2].shape =", a[:, :, ::2, ::2].shape)
print()

print("-----select by specific index-----")
print("a.shape =", a.shape)
print("a.index_select(0, torch.tensor([0, 2])).shape =\n", a.index_select(0, torch.tensor([0, 2])).shape)
print("a.index_select(1, torch.tensor([1, 2])).shape =\n", a.index_select(1, torch.tensor([1, 2])).shape)
print("a.index_select(2, torch.arange(28)).shape =", a.index_select(2, torch.arange(28)).shape)
print("a.index_select(2, torch.arange(8)).shape =", a.index_select(2, torch.arange(8)).shape)
print("a[...].shape =", a[...].shape)
print("a[0, ...].shape =", a[0, ...].shape)
print("a[:, 1, ...].shape =", a[:, 1, ...].shape)
print("a[..., :2].shape =", a[..., :2].shape)
print()

print("-----select by mask-----")
x = torch.randn(3, 4)
print("torch.randn(3, 4) =\n", x)
mask = x.ge(0.5)
print("mask =", mask)
print("torch.masked_select(x, mask) =", torch.masked_select(x, mask))
print("torch.masked_select(x, mask).shape =", torch.masked_select(x, mask).shape)
print()

print("-----select by flatten index-----")
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print("torch.tensor([[4, 3, 5], [6, 7, 8]]) =\n", src)
print("torch.take(src, torch.tensor([0, 2, 3])) =", torch.take(src, torch.tensor([0, 2, 5])))
