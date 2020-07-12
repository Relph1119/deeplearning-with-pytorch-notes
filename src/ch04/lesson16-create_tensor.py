#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson16-create_tensor.py
@time: 2020/7/12 9:35
@project: deeplearning-with-pytorch-notes
@desc: 第16课-创建Tensor
"""
import numpy as np
import torch

# Import from numpy
print("-----Import from numpy-----")
a = np.array([2, 3.3])
print("torch.from_numpy(np.array([2, 3.3])) =", torch.from_numpy(a))
a = np.ones([2, 3])
print("torch.from_numpy(np.ones([2, 3])) =", torch.from_numpy(a))
print()

print("-----Import from List-----")
print("torch.tensor([2., 3.2]) =", torch.tensor([2., 3.2]))
print("torch.FloatTensor([2., 3.2]) =", torch.FloatTensor([2., 3.2]))
print("torch.tensor([[2., 3.2], [1., 22.3]]) =\n", torch.tensor([[2., 3.2], [1., 22.3]]))
print()

print("-----uninitialized-----")
print("torch.empty(1) =", torch.empty(1))
print("torch.Tensor(2, 3) =\n", torch.Tensor(2, 3))
print("torch.IntTensor(2, 3) =\n", torch.IntTensor(2, 3))
print()

print("-----set default type-----")
print("torch.tensor([1.2, 3]).type() =", torch.tensor([1.2, 3]).type())
print("torch.tensor([1.2, 3]).type() =", torch.tensor([1.2, 3]).type())
print()

print("-----rand / rand_like, randint-----")
print("torch.rand(3, 3) =", torch.rand(3, 3))
a = torch.rand(3, 3)
print("torch.rand_like(torch.rand(3, 3)) =\n", torch.rand_like(a))
print("torch.randint(1, 10, (3, 3)) =\n", torch.randint(1, 10, (3, 3)))
print()

print("-----randn-----")
print("torch.randn(3, 3) =\n", torch.randn(3, 3))
a = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print("torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)) =\n", a)
b = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print("torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)) =\n", b)
print()

print("-----full-----")
print("torch.full([2, 3], 7) =\n", torch.full([2, 3], 7))
print("torch.full([], 7) =\n", torch.full([], 7))
print("torch.full([1], 7) =\n", torch.full([1], 7))
print()

print("-----arange / range-----")
print("torch.arange(0, 10) =\n", torch.arange(0, 10))
print("torch.arange(0, 10, 2) =\n", torch.arange(0, 10, 2))
print()

print("-----linspace / logspace-----")
print("torch.linspace(0, 10, steps=4) =\n", torch.linspace(0, 10, steps=4))
print("torch.linspace(0, 10, steps=10) =\n", torch.linspace(0, 10, steps=10))
print("torch.linspace(0, 10, steps=11) =\n", torch.linspace(0, 10, steps=11))
print("torch.logspace(0, -1, steps=10) =\n", torch.logspace(0, -1, steps=10))
print("torch.logspace(0, 1, steps=10) =\n", torch.logspace(0, 1, steps=10))
print()

print("-----ones / zeros / eye-----")
print("torch.ones(3, 3) =\n", torch.ones(3, 3))
print("torch.zeros(3, 3) =\n", torch.zeros(3, 3))
print("torch.eye(3, 4) =\n", torch.eye(3, 4))
print("torch.eye(3) =\n", torch.eye(3))
a = torch.zeros(3, 3)
print("torch.ones_like(a) =\n", torch.ones_like(a))
print()

print("-----randperm-----")
print("torch.randperm(10) =\n", torch.randperm(10))
a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print("idx =", idx)
print("idx =", idx)

print("a[idx] =", a[idx])
print("b[idx] =", b[idx])
print("a, b =", (a, b))
