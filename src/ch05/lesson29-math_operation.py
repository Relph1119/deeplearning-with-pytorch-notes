#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson29-math_operation.py
@time: 2020/7/14 17:07
@project: deeplearning-with-pytorch-notes
@desc: 第29-30课：数学运算
"""
import torch

print("-----Basic-----")
a = torch.rand(3, 4)
b = torch.rand(4)
print("a + b =", a + b)
print("torch.add(a, b) =", torch.add(a, b))
print("torch.all(torch.eq(a - b, torch.sub(a, b))) =", torch.all(torch.eq(a - b, torch.sub(a, b))))
print("torch.all(torch.eq(a * b, torch.mul(a, b))) =", torch.all(torch.eq(a * b, torch.mul(a, b))))
print("torch.all(torch.eq(a / b, torch.div(a, b))) =", torch.all(torch.eq(a / b, torch.div(a, b))))

# 矩阵相乘
print("-----matmul-----")
a = 3 * torch.ones(2, 2)
b = torch.ones(2, 2)
print("torch.mm(a, b) =\n", torch.mm(a, b))
print("torch.matmul(a, b) =\n", torch.matmul(a, b))
print("a @ b =\n", a @ b)
print()

a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
# .t()只适用于二维矩阵
print("(x @ w.t()).shape =", (x @ w.t()).shape)
print()

print("-----2d tensor matmul?-----")
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print("a.shape =", a.shape)
print("b.shape =", b.shape)
print("torch.matmul(a, b).shape =", torch.matmul(a, b).shape)
b = torch.rand(4, 1, 64, 32)
print("b.shape =", b.shape)
print("torch.matmul(a, b).shape =", torch.matmul(a, b).shape)
print()

print("-----power-----")
a = torch.full([2, 2], 3, dtype=torch.float)
print("a =\n", a)
print("a.pow(2) =\n", a.pow(2))
print("a ** 2 =\n", a ** 2)
aa = a ** 2
print("aa.sqrt() =\n", aa.sqrt())
print("aa.rsqrt() =\n", aa.rsqrt())
print("aa ** (0.5) =\n", aa ** (0.5))
print()

print("-----exp / log-----")
a = torch.exp(torch.ones(2, 2))
print("a =", a)
print("torch.log(a) =", torch.log(a))
print()

# 近似值
print("-----Approximation-----")
a = torch.tensor(3.14)
print("a =", a)
print("a.floor() =", a.floor(), "\na.ceil() =", a.ceil(), "\na.trunc() =", a.trunc(), "\na.frac() =", a.frac())
a = torch.tensor(3.499)
print("a =", a)
print("a.round() =", a.round())
a = torch.tensor(3.5)
print("a =", a)
print("a.round() =", a.round())
print()

print("-----clamp-----")
grad = torch.rand(2, 3) * 15
print("grad.max() =", grad.max())
print("grad.median() =", grad.median())
print("grad =\n", grad)
print("grad.clamp(10) =\n", grad.clamp(10))
print("grad.clamp(0, 10) =\n", grad.clamp(0, 10))