#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson14-tensor_type.py
@time: 2020/7/11 16:25
@project: deeplearning-with-pytorch-notes
@desc: 第14-15课 张量数据类型
"""
import numpy as np
import torch

print("-----Type Check-----")
a = torch.randn(2, 3)
print("a.type() =", a.type())
print("type(a) =", type(a))
print("isinstance(a, torch.FloatTensor) =", isinstance(a, torch.FloatTensor))

print("-----Dimension 0 / rank 0-----")
print("torch.tensor(1) =", torch.tensor(1))
print("torch.tensor(1.3) =", torch.tensor(1.3))
print()

print("-----Dim 0-----")
a = torch.tensor(2.2)
print("a.shape =", a.shape)
print("len(a.shape) =", len(a.shape))
print("a.size() =", a.size())
print()

print("-----Dim 1-----")
print("torch.tensor([1.1]) =", torch.tensor([1.1]))
print("torch.tensor([1.1, 2.2]) =", torch.tensor([1.1, 2.2]))
print("torch.FloatTensor(1) =", torch.FloatTensor(1))
print("torch.FloatTensor(2) =", torch.FloatTensor(2))
data = np.ones(2)
print("data =", data)
print("torch.from_numpy(data) =", torch.from_numpy(data))
a = torch.ones(2)
print("a.shape =", a.shape)
print()

print("-----Dim 2-----")
a = torch.randn(2, 3)
print("torch.randn(2, 3) =\n", a)
print("a.shape =", a.shape)
print("a.size(0) =", a.size(0))
print("a.size(1) =", a.size(1))
print("a.shape[1] =", a.shape[1])
print()

# Dim 3适用于onehot
print("-----Dim 3-----")
a = torch.rand(1, 2, 3)
print("torch.rand(1, 2, 3) =\n", a)
print("a.shape =", a.shape)
print("a[0] =", a[0])
print("list(a.shape) =", list(a.shape))
print()

print("-----Dim 4-----")
a = torch.rand(2, 3, 28, 28)
print("torch.rand(2 ,3, 28, 28) =\n", a)
print("a.shape =", a.shape)
print()

print("----Mixed-----")
print("a.shape =", a.shape)
print("a.numel() =", a.numel())
print("a.dim() =", a.dim())
a = torch.tensor(1)
print("a.dim() =", a.dim())
print()
