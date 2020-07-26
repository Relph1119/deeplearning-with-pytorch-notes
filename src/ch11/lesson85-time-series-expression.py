#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson85-time-series-expression.py
@time: 2020/7/24 14:13
@project: deeplearning-with-pytorch-notes
@desc: 第85课：时间序列表示
"""

import torch
from torch import nn
import torchnlp

from torchnlp import word_to_vector
from torchnlp.word_to_vector import GloVe

print("-----word2vec vs GloVe-----")
print("====word2vec=====")
word_to_ix = {"hello": 0, "world": 1}
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype=torch.long)

embeds = nn.Embedding(2, 5)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
print()

print("====GloVe=====")
vectors = GloVe()
print(vectors['hello'])
