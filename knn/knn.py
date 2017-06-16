#-*- coding: UTF-8 -*-
import numpy as np
import operator
import os
from os import listdir

# Doses begin
#读取文件为numpy数据，
def file2matrix(filename):
    fp = open(filename)
    lines = fp.readlines()
    num_of_lines = len(lines)
    ret_mat = np.zeros((num_of_lines, 3))
    label = []
    index = 0

    for line in lines:
        line = line.strip()
        l = line.split('\t')
        ret_mat[index] = l[0:3]
        label.append(l[-1])
        index += 1

    return ret_mat,label

#knn 分类
def classify0(in_x,dataset,labels,k):
    dataset_size = dataset.shape[0]
    diff_mat = in_x - dataset # numpy Broadcasting
    sq_mat = diff_mat**2
    sq_distance = sq_mat.sum(axis=1)
    distances = sq_distance**0.5
    sort_distance_index = distances.argsort()
    class_count = {}

    for i in range(k):
        cur_label = labels[sort_distance_index[i]]
        class_count[cur_label] = class_count.get(cur_label,0) + 1
        sort_class_cout = sorted(class_count.iteritems(),key = operator.itemgetter(1),reverse=True)
    
    return sort_class_cout[0][0]

#归一化
def normal(dataset):
    minval = dataset.min(0)
    maxval = dataset.max(0)
    ranges = maxval - minval
    normal_dataset = dataset -  minval
    normal_dataset = normal_dataset/ranges
    return normal_dataset
    
#测试
def knn_doses():
    dataset,labels = file2matrix(os.path.dirname(__file__) +'/datingTestSet.txt')
    dataset = normal(dataset)
    data_size = dataset.shape[0]
    test_data = dataset[0:data_size/10:1,0:]
    test_labels = labels[0:data_size/10:1]
    dataset = dataset[data_size/10:data_size:1,0:]
    labels = labels[data_size/10:data_size:1]
    index = 0
    true_count = 0
    for in_x in test_data:
        label = classify0(in_x,dataset,labels,5)
        if(label == test_labels[index]):
            true_count += 1
        
        index += 1
    
    print true_count,'/',index

# Doses end 
knn_doses()

