#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import tqdm
import torch
import dgl
import random
import wandb
import math
import time
import numpy as np
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta


def train(args, model, optimizer, device, graph_l, logger, n):

    best_param = {'best_mrr': 0, 'best_state': None, 'best_s_dw': None}
    earl_stop_c = 0
    epoch_count = 0

    for epoch in range(args.epochs):
        # Keep a version of the data without gradient calculation
        graph_l_cpy = deepcopy(graph_l)
        all_mrr = 0.0
        i = 0
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
        S_dw = [0] * len(fast_weights)
        train_count = 0
        # LightDyG windows calculation
        while i < (n - args.window_num):
            if i != 0:
                i = random.randint(i, i + args.window_num)
            if i >= (n - args.window_num):
                break
            graph_train = graph_l[i: i + args.window_num]
            i = i + 1
            # Copy a version of data as a valid in the window
            features = [graph_unit.node_feature.to(device) for graph_unit in graph_train]

            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            window_mrr = 0.0
            losses = torch.tensor(0.0).to(device)
            count = 0
            # one window
            for idx, graph in enumerate(graph_train):
                # The last snapshot in the window is valid only
                if idx == args.window_num - 1:
                    break
                # t snapshot train
                # Copy a version of data as a train in the window
                feature_train = deepcopy(features[idx])
                graph = graph.to(device)
                pred = model(graph, feature_train, fast_weights)
                loss = Link_loss_meta(pred, graph.edge_label)

                # t grad
                grad = torch.autograd.grad(loss, fast_weights)

                graph = graph.to('cpu')
                feature_train = feature_train.to('cpu')

                beta = args.beta
                S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))

                fast_weights = list(
                    map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

                # t+1 snapshot valid
                graph_train[idx + 1] = graph_train[idx + 1].to(device)
                pred = model(graph_train[idx + 1], features[idx + 1], fast_weights)
                loss = Link_loss_meta(pred, graph_train[idx + 1].edge_label)

                edge_label = graph_train[idx + 1].edge_label
                edge_label_index = graph_train[idx + 1].edge_label_index
                mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_train[idx + 1], features[idx+1],
                                                                  fast_weights)
                graph_train[idx + 1].edge_label = edge_label
                graph_train[idx + 1].edge_label_index = edge_label_index

                droprate = torch.FloatTensor(np.ones(shape=(1)) * args.drop_rate)
                masks = torch.bernoulli(1. - droprate).unsqueeze(1)
                if masks[0][0]:
                    losses = losses + loss
                    count += 1
                    window_mrr += mrr
                acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_train[idx + 1].edge_label)
                logger.info('meta epoch:{}, mrr:{:.5f}, loss: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                            format(epoch, mrr, loss, acc, ap, f1, macro_auc, micro_auc))

            if losses:
                losses = losses / count
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            if count:
                all_mrr += window_mrr / count
            train_count += 1

        all_mrr = all_mrr / train_count
        epoch_count += 1

        if all_mrr > best_param['best_mrr']:
            best_param = {'best_mrr': all_mrr, 'best_state': deepcopy(model.state_dict()), 'best_s_dw': S_dw}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    return best_param