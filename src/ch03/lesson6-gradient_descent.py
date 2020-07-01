#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson6-gradient_descent.py
@time: 2020/7/1 11:13
@project: deeplearning-with-pytorch-notes
@desc: 第6课-简单回归，采用梯度下降法
"""
import numpy as np


def compute_error_for_line_given_points(b, w, points):
    """
    计算average(sum{(wx+b-y)^2})
    :param b:
    :param w:
    :param points: 训练数据集
    :return:
    """
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    """
    更新梯度
    :param b_current:
    :param w_current:
    :param points:
    :param learning_rate:
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)
    return new_b, new_w


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """
    循环计算gradient
    :param points:
    :param starting_b:
    :param starting_w:
    :param learning_rate:
    :param num_iterations:
    :return:
    """
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return b, w


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    b, w = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()
