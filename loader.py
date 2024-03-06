import math

import networkx as nx
import time
import logging
import pickle
import numpy as np
import os
from deepsnap.graph import Graph

from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T

from graphgym.config import cfg
import graphgym.models.feature_augment as preprocess
from graphgym.models.transform import (ego_nets, remove_node_feature,
                                       edge_nets, path_len)
from graphgym.contrib.loader import *
import graphgym.register as register

from ogb.graphproppred import PygGraphPropPredDataset
from deepsnap.batch import Batch
from graphgym.contrib.loader.dataset_prep import load

import pdb


def load_pyg(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name,
                                    transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name, dataset_dir):
    '''
    load networkx format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except:
        graphs = nx.read_gpickle('{}.gpickle'.format(dataset_dir, name))
        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
    # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset):
    '''
    Dataset transformation before train/val/test split
    :param dataset: A DeepSNAP dataset object
    :return: A transformed DeepSNAP dataset object
    '''
    if cfg.dataset.remove_feature:
        dataset.apply_transform(remove_node_feature,
                                update_graph=True, update_tensor=False)
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
    if cfg.dataset.augment_label:
        dataset.apply_transform(preprocess._replace_label,
                                update_graph=True, update_tensor=False)
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    cfg.dataset.augment_feature_dims = actual_feat_dims
    if cfg.dataset.augment_label:
        cfg.dataset.augment_label_dims = actual_label_dim

    # Temporary for ID-GNN path prediction task
    if cfg.dataset.task == 'edge' and 'id' in cfg.gnn.layer_type:
        dataset.apply_transform(path_len, update_graph=False,
                                update_tensor=False)

    return dataset


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''
    if cfg.dataset.transform == 'ego':
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        cfg.dataset.task = 'node'
    return datasets


def create_dataset():
    ## Load dataset
    time1 = time.time()
    # 数据集是来自 pyg 还是 自己的
    if cfg.dataset.format in ['OGB']:
        graphs, splits = load_dataset()
    else:
        if cfg.dataset.format == 'dblp':
            dataset = 'dblp'
            e_feat = np.load('/share/share_40t/fangpeng/dataset/{0}/ml_{0}.npy'.format(dataset))
            n_feat = np.load('/share/share_40t/fangpeng/dataset/{0}/ml_{0}_node.npy'.format(dataset))
            e_time = np.load('/share/share_40t/fangpeng/dataset/{0}/ml_{0}_ts.npy'.format(dataset))
            train, train_e_feat, train_n_feat, test, test_e_feat, test_n_feat = load("Norandom", len(n_feat))
            n_feat = torch.Tensor(n_feat)
            e_feat = torch.Tensor(e_feat[:-1])
            # 将数据集全部组合起来
            graphs = []
            count = 0
            for tr in train:
                row = tr.row
                col = tr.col
                edge_index = np.vstack((row, col))
                edge_index = torch.Tensor(edge_index).long()
                e_time = [e_time[count] for i in range(edge_index.shape[1])]
                e_time = torch.Tensor(e_time)
                graphs.append(Graph(
                    node_feature=n_feat,
                    edge_feature=e_feat,
                    edge_index=edge_index,
                    edge_time=e_time,
                    directed=True
                ))
                count += 1

            for te in test:
                row = te.row
                col = te.col
                edge_index = np.vstack((row, col))
                edge_index = torch.Tensor(edge_index).long()
                e_time = [e_time[count] for i in range(edge_index.shape[1])]
                e_time = torch.Tensor(e_time)
                graphs.append(Graph(
                    node_feature=n_feat,
                    edge_feature=e_feat,
                    edge_index=edge_index,
                    edge_time=e_time,
                    directed=True
                ))
                count += 1
            for g_snapshot in graphs:
                g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
                g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
                g_snapshot.node_degree_existing = torch.zeros(n_feat.size(0))
        else:
            graphs = load_dataset()



    ## Filter graphs
    time2 = time.time()
    # n = math.ceil(len(graphs) * 0.7)
    # test_data = graphs[n-1:]
    # graphs = graphs[:n]
    min_node = filter_graphs()

    # 保存数据
    # for i, graph in enumerate(graphs):
    #     edge_index = graph.edge_index.numpy()
    #     edge_feature = graph.edge_feature.numpy()
    #     node_feature = graph.node_feature.numpy()
    #     edge_time = graph.edge_time.numpy()
    #
    #     path = "/share/share_40t/fangpeng/dataset/stackoverflow_M"
    #
    #     if not os.path.exists(path + '/' + 'edge_index'):
    #         os.makedirs(path + '/' + 'edge_index')
    #     path_ei = path + '/' + 'edge_index/' + str(i) + '.npy'
    #
    #     if not os.path.exists(path + '/' + 'node_feature'):
    #         os.makedirs(path + '/' + 'node_feature')
    #     path_nf = path + '/' + 'node_feature/' + str(i) + '.npy'
    #
    #     if not os.path.exists(path + '/' + 'edge_feature'):
    #         os.makedirs(path + '/' + 'edge_feature')
    #     path_ef = path + '/' + 'edge_feature/' + str(i) + '.npy'
    #
    #     if not os.path.exists(path + '/' + 'edge_time'):
    #         os.makedirs(path + '/' + 'edge_time')
    #     path_et = path + '/' + 'edge_time/' + str(i) + '.npy'
    #
    #     np.save(path_ei, edge_index)
    #     np.save(path_ef, edge_feature)
    #     np.save(path_nf, node_feature)
    #     np.save(path_et, edge_time)



    ## Create whole dataset

    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=min_node)
    # graphs = dataset.graphs[0].edge_label_index
    # graphs = graphs[:, graphs.size(1) // 2:]

    # test_data = GraphDataset(
    #     test_data,
    #     task=cfg.dataset.task,
    #     edge_train_mode=cfg.dataset.edge_train_mode,
    #     edge_message_ratio=cfg.dataset.edge_message_ratio,
    #     edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
    #     minimum_node_per_graph=min_node)

    ## Transform the whole dataset
    dataset = transform_before_split(dataset)
    # test_data = transform_before_split(test_data)

    ## Split dataset
    time3 = time.time()
    # Use custom data splits
    if cfg.dataset.split_method == 'chronological_temporal':
        if cfg.train.mode == 'live_update_fixed_split':
            datasets = [dataset, dataset, dataset]
        else:
            total = len(dataset)  # total number of snapshots.
            train_end = int(total * cfg.dataset.split[0])
            val_end = int(total * (cfg.dataset.split[0] + cfg.dataset.split[1]))
            datasets = [
                dataset[:train_end],
                dataset[train_end:val_end],
                dataset[val_end:]
            ]
    else:
        if cfg.dataset.format == 'OGB':
            datasets = []
            datasets.append(dataset[splits['train']])
            datasets.append(dataset[splits['valid']])
            datasets.append(dataset[splits['test']])
        # Use random split, supported by DeepSNAP
        else:

            # dataset = dataset.split(split_ratio=[0.7, 0.3], shuffle=False)
            datasets = dataset.split(
                transductive=cfg.dataset.transductive,
                split_ratio=cfg.dataset.split,
                shuffle=cfg.dataset.shuffle)
    ## Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)
    # test_data = transform_after_split(test_data)

    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
                 time2 - time1, time3 - time2, time4 - time3, time5 - time4))
    # return datasets, test_data
    return datasets


def create_loader(datasets):
    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders
