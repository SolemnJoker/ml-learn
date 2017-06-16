#-*- coding: UTF-8 -*-
from math import log

def calShannonEnt(dataset):
    data_size = len(dataset)
    label_count = {}
    for feat_vec in dataset:
        cur_label = feat_vec[-1]
        label_count[cur_label] = label_count.get(cur_label,0) + 1
        shannon_ent = 0.0
    for key in label_count:
        prob = label_count[key]/data_size
        shannon_ent -= prob* log(prob,2)
    return shannon_ent
