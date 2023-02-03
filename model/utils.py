#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import dgl
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from model.config import cfg
from torch_scatter import scatter_max, scatter_mean, scatter_min
from model.loss import prediction, Link_loss_meta


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def create_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "softmax":
        return nn.Softmax()

    raise RuntimeError("activation error, not {}".format(activation))


def edge_index_difference(edge_all, edge_except, num_nodes):

    idx_all = edge_all[0] * num_nodes + edge_all[1]
    idx_except = edge_except[0] * num_nodes + edge_except[1]
    # filter out edges in idx_except.
    mask = torch.from_numpy(np.isin(idx_all, idx_except)).to(torch.bool)
    # mask = torch.isin(idx_all, idx_except)
    idx_kept = idx_all[~mask]
    i = idx_kept // num_nodes
    j = idx_kept % num_nodes
    return torch.stack([i, j], dim=0).long()


def gen_negative_edges(edge_index, num_neg_per_node, num_nodes):
    src_lst = torch.unique(edge_index[0])  # get unique senders.
    num_neg_per_node = int(1.5 * num_neg_per_node)  # add some redundancy.
    i = src_lst.repeat_interleave(num_neg_per_node)
    # nodes = torch.unique(edge_index.flatten())
    # nodes = nodes.cpu().numpy()
    j = torch.Tensor(np.random.choice(num_nodes, len(i), replace=True))
    # candidates for negative edges, X candidates from each src.
    candidates = torch.stack([i, j], dim=0).long()

    # filter out positive edges in candidate.
    neg_edge_index = edge_index_difference(candidates, edge_index.to('cpu'),
                                           num_nodes)
    return neg_edge_index


@torch.no_grad()
def fast_batch_mrr_and_recall(edge_label_index, edge_label, pred_score, num_neg_per_node, num_nodes):

    src_lst = torch.unique(edge_label_index[0], sorted=True)
    num_users = len(src_lst)

    edge_pos = edge_label_index[:, edge_label == 1]
    edge_neg = edge_label_index[:, edge_label == 0]

    # By construction, negative edge index should be sorted by their src nodes.
    assert torch.all(edge_neg[0].sort()[0] == edge_neg[0])

    # Prediction scores of all positive and negative edges.
    p_pos = pred_score[edge_label == 1]
    p_neg = pred_score[edge_label == 0]

    # For each player src, compute the highest score among all positive edges
    # from src.
    # We want to compute the rank of this edge.
    # Construct an interval of model's performance.
    if cfg.metric.mrr_method == 'mean':
        best_p_pos = scatter_mean(src=p_pos, index=edge_pos[0],
                                  dim_size=num_nodes)
    elif cfg.metric.mrr_method == 'min':
        best_p_pos, _ = scatter_min(src=p_pos, index=edge_pos[0],
                                    dim_size=num_nodes)
    else:
        # The default setting, consider the rank of the most confident edge.
        best_p_pos, _ = scatter_max(src=p_pos, index=edge_pos[0],
                                    dim_size=num_nodes)
    # best_p_pos has shape (num_nodes), for nodes not in src_lst has value 0.
    # 取出了节点上的最大概率
    best_p_pos_by_user = best_p_pos[src_lst]

    # Sanity check.
    # src_lst_2, inverse = torch.unique(edge_pos[0], return_inverse=True)
    # best_p_pos, _ = scatter_max(p_pos, inverse)
    # assert torch.all(best_p_pos_by_user == best_p_pos)

    uni, counts = torch.unique(edge_neg[0], sorted=True, return_counts=True)
    # assert torch.all(counts >= num_neg_per_node)
    # assert torch.all(uni == src_lst)
    # note: edge_neg (src, dst) are sorted by src.
    # find index of first occurrence of each src in edge_neg[0].
    # neg edges[0], [1,1,...1, 2, 2, ... 2, 3, ..]
    first_occ_idx = torch.cumsum(counts, dim=0) - counts
    add = torch.arange(num_neg_per_node, device=first_occ_idx.device)

    # take the first 100 negative edges from each src.(取前一百个负样本)
    score_idx = first_occ_idx.view(-1, 1) + add.view(1, -1)

    assert torch.all(edge_neg[0][score_idx].float().std(axis=1) == 0)
    # Z = edge_neg[0][first_occ_idx - 1]
    # A = edge_neg[0][first_occ_idx]
    # B = edge_neg[0][first_occ_idx + 1]
    # assert torch.all(Z != A)
    # assert torch.all(B == A)
    # 前一百个负样本的预测
    p_neg_by_user = p_neg[score_idx]  # (num_users, num_neg_per_node)
    # 比较100个负样本和正样本最大概率
    compare = (p_neg_by_user >= best_p_pos_by_user.view(num_users, 1)).float()
    assert compare.shape == (num_users, num_neg_per_node)
    # compare[i, j], for node i, the j-th negative edge's score > p_best.

    # counts 1 + how many negative edge from src has higher score than p_best.
    # if there's no such negative edge, rank is 1.
    # 所有src的负样本大于正样本的统计
    rank_by_user = compare.sum(axis=1) + 1  # (num_users,)
    assert rank_by_user.shape == (num_users,)

    mrr = float(torch.mean(1 / rank_by_user))
    # print(f'MRR={mrr}, time taken: {datetime.now() - start}')
    # computes recall at k as well
    recall_at = dict()
    for k in [1, 3, 10]:
        recall_at[k] = float((rank_by_user <= k).float().mean())

    return mrr, recall_at


@torch.no_grad()
def report_rank_based_eval_meta(model, graph, x, fast_weights, num_neg_per_node: int = 1000):
    if num_neg_per_node == -1:
        # Do not report rank-based metrics, used in debug mode.
        return 0, 0, 0, 0
    # Get positive edge indices.
    edge_index = graph.edge_label_index[:, graph.edge_label == 1]

    neg_edge_index = gen_negative_edges(edge_index, num_neg_per_node, num_nodes=graph.num_nodes())

    new_edge_label_index = torch.cat((edge_index, neg_edge_index), dim=1)
    new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
                                torch.zeros(neg_edge_index.shape[1])
                                ), dim=0)

    # Construct evaluation samples.
    graph.edge_label_index = new_edge_label_index.to('cpu').long()
    graph.edge_label = new_edge_label.to('cpu').long()

    # move state to gpu
    pred = model(graph, x, fast_weights)
    pred = pred.to('cpu')

    mrr, recall_at = fast_batch_mrr_and_recall(graph.edge_label_index, graph.edge_label,
                                               pred, num_neg_per_node, graph.num_nodes())

    return mrr, recall_at[1], recall_at[3], recall_at[10]


def rand_prop(graph):
    features = graph.node_feature
    n = features.shape[0]
    # mask
    drop_rate = cfg.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(features)

    features = masks * features

    return features


def update_states(states, fast_weights):
    count = 0
    for key in states.keys():
        assert isinstance(states[key], torch.Tensor)
        states[key] = fast_weights[count]
        count += 1
    return states


def paramters_(state):
    out = list()
    for key in state.keys():
        state[key].requires_grad=True
        out.append(state[key])
    return out
