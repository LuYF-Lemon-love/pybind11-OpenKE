# coding:utf-8
#
# pybind11_ke/data/GraphTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
#
# RGCN 的测试数据采样器.

import torch
import numpy as np
from .GraphSampler import GraphSampler
from collections import defaultdict as ddict

class GraphTestSampler(object):

    """为测试进行图采样
    """

    def __init__(
        self,
        sampler: GraphSampler):

        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.ent_tol = sampler.ent_tol
        self.triples = sampler.train_triples

    def get_hr2t_rt2h_from_all(self):

        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        
        batch_data = {}
        head_label = torch.zeros(len(data), self.ent_tol)
        tail_label = torch.zeros(len(data), self.ent_tol)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        head, rela, tail = np.array(self.triples).transpose()
        graph, rela, norm = self.sampler.build_graph(self.ent_tol, (head, rela, tail), -1)
        batch_data["graph"]  = graph
        batch_data["rela"]   = rela
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.ent_tol, dtype=torch.long).view(-1,1)
        
        return batch_data