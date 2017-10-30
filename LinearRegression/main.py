# -*- coding: utf-8 -*-
import os

import xlrd
import xlwt
import sys

import operator

# import numpy as np
# from numpy import *

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

class data_stucture:
	def __init__(self, score, hour):
		self.score = score
		self.hour = hour

def data_analyse():
    file_txt = open("original.txt", "r")
    data_list = []
    count = 0
    for line in file_txt:

        temp_list = line.strip().split(',')
        # print(temp_list[0])
        # print(temp_list[1])
        data_list.append(data_stucture(temp_list[0], temp_list[1]))

    file_txt.close()

    target_data = xlwt.Workbook(encoding="utf-8")
    target_sheet = target_data.add_sheet(u'sheet1')

    for item in data_list:
        # print(item.score)
        # print(item.hour)
        target_sheet.write(count, 0, item.score)
        target_sheet.write(count, 1, item.hour)
        count = count + 1

    target_data.save("data.xls")
    print("count = ", count)

    return data_list

def run():
    # points = genfromtext('original.txt', delimiter=',', dtype='f8,U2')
    points = data_analyse()
    # print(points)
    learning_rate = 0.0001
    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0  # ideal slope, will start with 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error ={3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

# the gradient_descent_runner function is defined below:
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for point in points:
        x = float(point.score)
        y = float(point.hour)
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def compute_error_for_line_given_points(b, m, points):
    totalError = 0

    # print(b)
    # print(m)
    # print(float(len(points)))

    for point in points:
        x = float(point.score)
        y = float(point.hour)
        totalError += (y - (m * x + b)) ** 2

    return(totalError / float(len(points)))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]

if __name__=="__main__":
    print("== start ==")

    # data_list = []
    # data_list = data_analyse()
    run()

    print("== end ==")