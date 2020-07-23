#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson56-cross-validation.py
@time: 2020/7/23 14:33
@project: deeplearning-with-pytorch-notes
@desc: 第56-57课：交叉验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def load_data(batch_size):
    train_db = datasets.MNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    test_db = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=batch_size, shuffle=True)

    train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_db,
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


def training(train_loader, net, device, optimizer, criteon):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def validating(val_loader, net, device, criteon):
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def testing(test_loader, net, device, criteon):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


global net

if __name__ == '__main__':
    batch_size = 200
    learning_rate = 0.01
    epochs = 10

    train_loader, test_loader, val_loader = load_data(batch_size)
    print('train:', len(train_loader.dataset), 'test:', len(test_loader.dataset), 'val:', len(val_loader.dataset))

    device = torch.device('cuda:0')
    net = MLP().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        training(train_loader, net, device, optimizer, criteon)
        validating(val_loader, net, device, criteon)

    testing(test_loader, net, device, criteon)
