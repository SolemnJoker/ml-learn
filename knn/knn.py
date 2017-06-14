import numpy as np
import operator
from os import listdir


def createDataset():
    group = array([1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1])
    lable = ['A', 'A', 'B', 'B']
    return group, lable


def file2matrix(filename):
    fp = open(filename)
    lines = fp.readlines()
    num_of_lines = len(lines)
    ret_mat = np.zeros((num_of_lines, 3))
    for line in lines:
        line = line.strip()
        l = line.split()
