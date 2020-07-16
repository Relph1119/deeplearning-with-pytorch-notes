#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson46-2D_func_optimization.py
@time: 2020/7/16 9:08
@project: deeplearning-with-pytorch-notes
@desc: 第46课：2D函数优化实例
"""
import numpy as np
from matplotlib import pyplot as plt
import torch


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('rainbow_r'))
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([-4., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.tolist(), pred.item()))
