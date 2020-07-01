#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson3-main.py
@time: 2020/7/1 9:27
@project: deeplearning-with-pytorch-notes
@desc: 第3课-测试Pytorch环境
"""

import torch

print(torch.__version__)
print('gpu:', torch.cuda.is_available())
