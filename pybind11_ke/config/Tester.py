"""

TrainDataLoader.py API.

TrainDataLoader.py - 通过 pybind11 与底层 C++ 数据处理模块交互。

.. code-block:: python

    # Import TrainDataLoader
    from pybind11_ke.data import TrainDataLoader

    # dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "./benchmarks/FB15K237/", 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 25,
		neg_rel = 0)
"""

# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
# from ..release import base
import base

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain = False):
        base.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            base.testHead(score, index, type_constrain)
            score = self.test_one_step(data_tail)
            base.testTail(score, index, type_constrain)
        base.test_link_prediction(type_constrain)

        mrr = base.getTestLinkMRR(type_constrain)
        mr = base.getTestLinkMR(type_constrain)
        hit10 = base.getTestLinkHit10(type_constrain)
        hit3 = base.getTestLinkHit3(type_constrain)
        hit1 = base.getTestLinkHit1(type_constrain)
        # print(hit10)
        return mrr, mr, hit10, hit3, hit1

    # def get_best_threshlod(self, score, ans):
    #     res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
    #     order = np.argsort(score)
    #     res = res[order]

    #     total_all = (float)(len(score))
    #     total_current = 0.0
    #     total_true = np.sum(ans)
    #     total_false = total_all - total_true

    #     res_mx = 0.0
    #     threshlod = None
    #     for index, [ans, score] in enumerate(res):
    #         if ans == 1:
    #             total_current += 1.0
    #         res_current = (2 * total_current + total_false - index - 1) / total_all
    #         if res_current > res_mx:
    #             res_mx = res_current
    #             threshlod = score
    #     return threshlod, res_mx

    # def run_triple_classification(self, threshlod = None):
    #     self.lib.initTest()
    #     self.data_loader.set_sampling_mode('classification')
    #     score = []
    #     ans = []
    #     training_range = tqdm(self.data_loader)
    #     for index, [pos_ins, neg_ins] in enumerate(training_range):
    #         res_pos = self.test_one_step(pos_ins)
    #         ans = ans + [1 for i in range(len(res_pos))]
    #         score.append(res_pos)

    #         res_neg = self.test_one_step(neg_ins)
    #         ans = ans + [0 for i in range(len(res_pos))]
    #         score.append(res_neg)

    #     score = np.concatenate(score, axis = -1)
    #     ans = np.array(ans)

    #     if threshlod == None:
    #         threshlod, _ = self.get_best_threshlod(score, ans)

    #     res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
    #     order = np.argsort(score)
    #     res = res[order]

    #     total_all = (float)(len(score))
    #     total_current = 0.0
    #     total_true = np.sum(ans)
    #     total_false = total_all - total_true

    #     for index, [ans, score] in enumerate(res):
    #         if score > threshlod:
    #             acc = (2 * total_current + total_false - index) / total_all
    #             break
    #         elif ans == 1:
    #             total_current += 1.0

    #     return acc, threshlod