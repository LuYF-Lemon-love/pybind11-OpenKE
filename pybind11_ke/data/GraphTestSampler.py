# coding:utf-8
#
# pybind11_ke/data/GraphTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 17, 2024
#
# R-GCN 的测试数据采样器.

"""
GraphTestSampler - R-GCN 的测试数据采样器。
"""

import dgl
import torch
import typing
import numpy as np
from .GraphSampler import GraphSampler
from .CompGCNSampler import CompGCNSampler
from collections import defaultdict as ddict

class GraphTestSampler(object):

    """``R-GCN`` :cite:`R-GCN` 的测试数据采样器。
    """

    def __init__(
        self,
        sampler: typing.Union[GraphSampler, CompGCNSampler]):

        """创建 GraphTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: typing.Union[GraphSampler, CompGCNSampler]
        """

        #: 训练数据采样器
        self.sampler: typing.Union[GraphSampler, CompGCNSampler] = sampler
        #: 实体的个数
        self.ent_tol: int = sampler.ent_tol
        #: 训练集三元组
        self.triples: list[tuple[int, int, int]] = sampler.train_triples
        #: 幂
        self.power: float = -1

        #: 知识图谱中所有 h-r 对对应的 t 集合
        self.hr2t_all: ddict[set] = ddict(set)
        #: 知识图谱中所有 r-t 对对应的 h 集合
        self.rt2h_all: ddict[set] = ddict(set)

        self.get_hr2t_rt2h_from_all()

    def get_hr2t_rt2h_from_all(self):

        """获得 :py:attr:`hr2t_all` 和 :py:attr:`rt2h_all` 。"""

        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(
        self,
        data: list[tuple[int, int, int]]) -> dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]:

        """``R-GCN`` :cite:`R-GCN` 的测试数据采样函数。
        
        :param data: 测试的正确三元组
        :type data: list[tuple[int, int, int]]
        :returns: ``R-GCN`` :cite:`R-GCN` 的测试数据
        :rtype: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]
        """
        
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
        
        graph, rela, norm = self.sampler.build_graph(self.ent_tol, np.array(self.triples).transpose(), self.power)
        batch_data["graph"]  = graph
        batch_data["rela"]   = rela
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.ent_tol, dtype=torch.long).view(-1,1)
        
        return batch_data