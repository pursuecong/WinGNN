#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
from model.config import cfg
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

def prediction(pred_score, true_l):
    # Acc_torch = torchmetrics.Accuracy(task='binary').to(pred_score)
    # Macro_Auc_torch = torchmetrics.AUROC(task='binary', average='macro').to(pred_score)
    # Micro_Auc_torch = torchmetrics.AUROC(task='binary', average='micro').to(pred_score)
    # Ap_torch = torchmetrics.AveragePrecision(task='binary').to(pred_score)
    # F1_torch = torchmetrics.F1Score(task='binary', average='macro').to(pred_score)
    #
    # acc_torch = Acc_torch(pred_score, true_l.to(pred_score))
    # macro_auc_torch = Macro_Auc_torch(pred_score, true_l.to(pred_score))
    # micro_auc_torch = Micro_Auc_torch(pred_score, true_l.to(pred_score))
    # ap_torch = Ap_torch(pred_score, true_l.to(pred_score))
    # f1_torch = F1_torch(pred_score, true_l.to(pred_score))

    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()

    # true = np.ones_like(pred)
    true = true_l
    true = true.cpu().numpy()
    acc = accuracy_score(true, pred)
    ap = average_precision_score(true, pred_score)
    f1 = f1_score(true, pred, average='macro')
    macro_auc = roc_auc_score(true, pred_score, average='macro')
    micro_auc = roc_auc_score(true, pred_score, average='micro')

    # print(acc, ap, f1, macro_auc, micro_auc)
    # print(acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch)
    return acc, ap, f1, macro_auc, micro_auc
    # return acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch


def Link_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)

    return loss

