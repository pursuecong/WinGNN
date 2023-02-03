#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :load datasets

import os
import copy
import math
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def load(nodes_num):
    """
    load_dataset
    :param nodes_num:
    :return:
    """
    path = "dataset/dblp_timestamp/"

    train_e_feat_path = path + 'train_e_feat/' + type + '/'
    test_e_feat_path = path + 'test_e_feat/' + type + '/'

    train_n_feat_path = path + type + '/' + 'train_n_feat/'
    test_n_feat_path = path + type + '/' + 'test_n_feat/'


    path = path + type
    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)
    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat


def load_r(name):
    path = "dataset/" + name
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'

    edge_index = read_npz(path_ei)
    edge_feature = read_npz(path_ef)
    node_feature = read_npz(path_nf)
    edge_time = read_npz(path_et)

    nodes_num = node_feature[0].shape[0]

    sub_graph = []
    for e_i in edge_index:
        row = e_i[0]
        col = e_i[1]
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph, edge_feature, edge_time, node_feature


def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path+filename))

    return npz


def read_e_feat(path):
    filesname = os.listdir(path)
    e_feat = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        e_feat.append(np.load(path+filename))

    return e_feat


def read_graph(path, nodes_num, num):

    filesname = os.listdir(path)
    # 对文件名做一个排序
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id) - num
        file_s[id] = filename

    # 文件读取
    sub_graph = []
    for file in file_s:
        sub_ = pd.read_csv(path + file)

        row = sub_.src_l.values
        col = sub_.dst_l.values

        node_m = set(row).union(set(col))
        # ts = torch.Tensor(sub_.timestamp.values)
        ts = [1] * len(row)

        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph


