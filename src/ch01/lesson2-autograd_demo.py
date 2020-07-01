#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson2-autograd_demo.py
@time: 2020/7/1 9:21
@project: deeplearning-with-pytorch-notes
@desc: 第2课-自动求导
"""

import torch
from torch import autograd

x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a ** 2 * x + b * x + c

print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after :', grads[0], grads[1], grads[2])
