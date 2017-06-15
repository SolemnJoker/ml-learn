import numpy as np
import operator
import os
from os import listdir

# Doses begin
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

def classify0(in_x,dataset,labels,k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(in_x,(dataset_size,1)) - dataset
    sq_mat = diff_mat**2
    sq_distance = sq_mat.sum(axis=1)
    distances = sq_distance**0.5
    sort_distance_index = distances.argsort()
    class_count = {}

    for i in range(k):
        cur_label = labels[sort_distance_index[i]]
        print type(cur_label),cur_label
        class_count[cur_label] = class_count.get(cur_label,0) + 1
        sort_class_cout = sorted(class_count.iteritems(),key = operator.itemgetter(1),reverse=True)
    
    return sort_class_cout[0][0]
    
def knn_doses():
    dataset,labels = file2matrix(os.path.dirname(__file__) +'/datingTestSet.txt')
    data_size = dataset.shape[0]
    test_data = dataset[0:data_size - 1:10,0:]
    test_labels = labels[0:data_size - 1:10]
    index = 0
    true_count = 0
    for in_x in test_data:
        label = classify0(in_x,dataset,labels,7)
        if(label == test_labels[index]):
            true_count += 1
        
        index += 1
    
    print true_count,'/',index

# Doses end 
knn_doses()

