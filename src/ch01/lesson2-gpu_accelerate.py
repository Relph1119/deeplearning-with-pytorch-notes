#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson2-gpu_accelerate.py
@time: 2020/7/1 9:14
@project: deeplearning-with-pytorch-notes
@desc: 第2课-测试GPU加速的矩阵乘法
"""

import time

import torch

print('pytorch version:', torch.__version__)
print('GPU是否可用:', torch.cuda.is_available())

a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
# 在CPU上运算矩阵乘法
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
# 在GPU上运行矩阵乘法
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

t0 = time.time()
# 确保运算的准确性
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
