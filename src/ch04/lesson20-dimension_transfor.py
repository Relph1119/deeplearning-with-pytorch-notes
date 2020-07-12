#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson20-dimension_transfor.py
@time: 2020/7/12 10:38
@project: deeplearning-with-pytorch-notes
@desc: 第20-21课：维度变换
"""
import torch

print("-----view / reshape-----")
a = torch.rand(4, 1, 28, 28)
print("a.shape =", a.shape)
print("a.view(4, 28 * 28) =\n", a.view(4, 28 * 28))
print("a.view(4, 28*28).shape =", a.view(4, 28 * 28).shape)
print("a.view(4*28, 28).shape = ", a.view(4 * 28, 28).shape)
print("a.view(4*1, 28, 28).shape =", a.view(4 * 1, 28, 28).shape)
b = a.view(4, 784)
b.view(4, 28, 28, 1)
print()

# [-a.dim-1, a.dim()+1)
print("-----unsqueeze-----")
print("a.shape =", a.shape)
print("a.unsqueeze(0).shape =", a.unsqueeze(0).shape)
print("a.unsqueeze(-1).shape =", a.unsqueeze(-1).shape)
print("a.unsqueeze(4).shape =", a.unsqueeze(4).shape)
print("a.unsqueeze(-4).shape =", a.unsqueeze(-4).shape)
print("a.unsqueeze(-5).shape =", a.unsqueeze(-5).shape)

a = torch.tensor([1.2, 2.3])
print("a.unsqueeze(-1) =\n", a.unsqueeze(-1))
print("a.unsqueeze(0) =", a.unsqueeze(0))
print()

print("-----For Example-----")
b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print("b.shape =", b.shape)
print()

print("-----squeeze-----")
print("b.shape =", b.shape)
print("b.squeeze().shape =", b.squeeze().shape)
print("b.squeeze(0).shape =", b.squeeze(0).shape)
print("b.squeeze(-1).shape =", b.squeeze(-1).shape)
print("b.squeeze(1).shape =", b.squeeze(1).shape)
print("b.squeeze(-4).shape =", b.squeeze(-4).shape)
print()

print("-----expand-----")
a = torch.rand(4, 3, 14, 14)
print("b.shape =", b.shape)
print("b.expand(4, 32, 14, 14).shape =", b.expand(4, 32, 14, 14).shape)
# 如果不想原来的维度不变，写-1
print("b.expand(-1, 32, -1, -1).shape =", b.expand(-1, 32, -1, -1).shape)
print("b.expand(-1, 32, -1, -4).shape =", b.expand(-1, 32, -1, -4).shape)
print()

print("-----repeat-----")
print("b.shape =", b.shape)
print("b.repeat(4, 32, 1, 1).shape =", b.repeat(4, 32, 1, 1).shape)
print("b.repeat(4, 1, 1, 1).shape =", b.repeat(4, 1, 1, 1).shape)
print("b.repeat(4, 1, 32, 32).shape =", b.repeat(4, 1, 32, 32).shape)
print()

print("-----transpose-----")
a = torch.randn(3, 4)
print("a.t() =\n", a.t())

a = torch.randn(4, 3, 32, 32)
print("a.shape =", a.shape)
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
print("a1.shape, a2.shape =", (a1.shape, a2.shape))
print("torch.all(torch.eq(a, a1)) =", torch.all(torch.eq(a, a1)))
print("torch.all(torch.eq(a, a2)) =", torch.all(torch.eq(a, a2)))
print()

print("-----permute-----")
a = torch.rand(4, 3, 28, 28)
print("a.transpose(1, 3).shape =", a.transpose(1, 3).shape)
b = torch.rand(4, 3, 28, 32)
print("b.transpose(1, 3).shape =", b.transpose(1, 3).shape)
print("b.transpose(1, 3).transpose(1, 2).shape =", b.transpose(1, 3).transpose(1, 2).shape)
print("b.permute(0, 2, 3, 1).shape =", b.permute(0, 2, 3, 1).shape)
