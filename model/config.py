#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import logging
import os
from yacs.config import CfgNode as CN

cfg = CN()

def set_cfg(cfg):
    r'''
    This function sets the default config value.

    :return: configuration use by the experiment.
    '''
    cfg.log_path = 'model_log/con_log'
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    # Link_prediction Edge_decoding
    cfg.model.edge_decoding = 'dot'

    # ------------------------------------------------------------------------ #
    # GNN options
    # ------------------------------------------------------------------------ #
    cfg.gnn = CN()
    # GNN skip connection
    cfg.gnn.skip_connection = 'affine'

    cfg.metric = CN()
    cfg.metric.mrr_method = 'max'


set_cfg(cfg)