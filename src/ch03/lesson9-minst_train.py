#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson9-minst_train.py
@time: 2020/7/1 13:47
@project: deeplearning-with-pytorch-notes
@desc: 第9-13课-MINST手写数字识别
"""

import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

from src.ch03.util import one_hot, plot_curve, plot_image


def load_dataset():
    batch_size = 512

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # xw + b
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1 + b)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2 + b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3 + b3
        x = self.fc3(x)

        return x


def train(epoch_num, train_loader, net, optimizer):
    """
    训练模型
    :param epoch_num: 迭代次数
    :param train_loader: 训练数据集
    :param net: 网络
    :param optimizer: 优化器
    :return:
    """
    train_loss = []
    for epoch in range(epoch_num):
        for batch_idx, (x, y) in enumerate(train_loader):
            # x: [b, 1, 28, 28], y: [512]
            # [b, 1, 28, 28] => [b, feature]
            x = x.view(x.size(0), 28 * 28)
            # => [b, 10]
            out = net(x)
            y_onehot = one_hot(y)
            # loss = mse(out, y_onehot)
            loss = F.mse_loss(out, y_onehot)

            optimizer.zero_grad()
            loss.backward()
            # w' = w - lr * grad
            optimizer.step()

            train_loss.append(loss.item())

            if batch_idx % 10 == 0:
                print(epoch, batch_idx, loss.item())

    return train_loss


if __name__ == '__main__':
    train_loader, test_loader = load_dataset()
    x, y = next(iter(train_loader))
    # print(x.shape, y.shape, x.min(), x.max())
    # plot_image(x, y, "image sample")

    net = Net()
    # [w1, b1, w2, b2, w3, b3]
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    epoch_num = 3
    train_loss = train(epoch_num, train_loader, net, optimizer)

    plot_curve(train_loss)
    # 得到参数[w1, b1, w2, b2, w3, b3]

    total_correct = 0
    for x, y in test_loader:
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        # out: [b, 10] => pred: [b]
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct

    total_num = len(test_loader.dataset)
    acc = total_correct / total_num
    print('test acc:', acc)

    x, y = next(iter(test_loader))
    out = net(x.view(x.size(0), 28 * 28))
    pred = out.argmax(dim=1)
    plot_image(x, pred, 'test')
