#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import numpy as np
import torch
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta


def test(graph_l, model, args, logger, n, S_dw, device):
    beta = args.beta
    avg_mrr = 0.0
    avg_auc = 0.0

    graph_test = graph_l[n:]
    # model parameters
    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    for idx, g_test in enumerate(graph_test):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)

        pred = model(g_test, graph_train, fast_weights)
        loss = Link_loss_meta(pred, g_test.edge_label)

        graph_train = graph_train.to('cpu')
        grad = torch.autograd.grad(loss, fast_weights)
        g_test = g_test.to('cpu')

        S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0].pow(2), zip(grad, S_dw)))

        fast_weights = list(map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

        graph_test[idx + 1] = graph_test[idx + 1].to(device)
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature.to(device)
        pred = model(graph_test[idx + 1], graph_test[idx + 1].node_feature, fast_weights)

        loss = Link_loss_meta(pred, graph_test[idx + 1].edge_label)

        edge_label = graph_test[idx + 1].edge_label
        edge_label_index = graph_test[idx + 1].edge_label_index
        mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                                          fast_weights)
        graph_test[idx + 1].edge_label = edge_label
        graph_test[idx + 1].edge_label_index = edge_label_index

        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx + 1].edge_label)
        avg_mrr += mrr
        avg_auc += macro_auc
        logger.info('meta test, mrr: {:.5f}, rl1: {:.5f}, rl3: {:.5f}, rl10: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                    format(mrr, rl1, rl3, rl10, acc, ap, f1, macro_auc, micro_auc))

    avg_mrr /= len(graph_test) - 1
    avg_auc /= len(graph_test) - 1
    logger.info({'avg_mrr': avg_mrr, 'avg_auc': avg_auc})
    return avg_mrr

