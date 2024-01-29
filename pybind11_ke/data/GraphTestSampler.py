# coding:utf-8
#
# pybind11_ke/data/GraphTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# R-GCN 的测试数据采样器.

"""
GraphTestSampler - R-GCN 的测试数据采样器。
"""

import os
import dgl
import torch
import typing
import numpy as np
from .TestSampler import TestSampler
from .GraphSampler import GraphSampler
from .CompGCNSampler import CompGCNSampler
from typing_extensions import override

class GraphTestSampler(TestSampler):

    """``R-GCN`` :cite:`R-GCN` 的测试数据采样器。

    例子::

        from pybind11_ke.data import GraphTestSampler, CompGCNTestSampler
        from torch.utils.data import DataLoader

        #: 测试数据采样器
        test_sampler: typing.Union[typing.Type[GraphTestSampler], typing.Type[CompGCNTestSampler]] = test_sampler(
            sampler=train_sampler,
            valid_file=valid_file,
            test_file=test_file,
        )
    
        #: 验证集三元组
        data_val: list[tuple[int, int, int]] = test_sampler.get_valid()
        #: 测试集三元组
        data_test: list[tuple[int, int, int]] = test_sampler.get_test()

        val_dataloader = DataLoader(
            data_val,
            shuffle=False,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=test_sampler.sampling,
        )

        test_dataloader = DataLoader(
            data_test,
            shuffle=False,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=test_sampler.sampling,
        )
    """

    def __init__(
        self,
        sampler: typing.Union[GraphSampler, CompGCNSampler],
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt"):

        """创建 GraphTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: typing.Union[GraphSampler, CompGCNSampler]
        :param valid_file: valid2id.txt
        :type valid_file: str
        :param test_file: test2id.txt
        :type test_file: str
        """

        super().__init__(
            sampler=sampler,
            valid_file=valid_file,
            test_file=test_file
        )

        #: 训练集三元组
        self.triples: list[tuple[int, int, int]] = self.sampler.t_triples if isinstance(self.sampler, CompGCNSampler) else self.sampler.train_triples
        #: 幂
        self.power: float = -1

        self.add_valid_test_reverse_triples()
        self.get_hr2t_rt2h_from_all()

    @override
    def get_valid_test_triples_id(self):

        """读取 :py:attr:`valid_file` 文件和 :py:attr:`test_file` 文件。"""
                
        with open(os.path.join(self.sampler.in_path, self.valid_file)) as f:
            self.valid_tol = (int)(f.readline())
            for line in f:
                h, t, r = line.strip().split()
                self.valid_triples.append((int(h), int(r), int(t)))
                
        with open(os.path.join(self.sampler.in_path, self.test_file)) as f:
            self.test_tol = (int)(f.readline())
            for line in f:
                h, t, r = line.strip().split()
                self.test_triples.append((int(h), int(r), int(t)))

    def add_valid_test_reverse_triples(self):

        """对于每一个三元组 (h, r, t)，生成相反关系三元组 (t, r`, h): r` = r + rel_tol。"""

        tol = int(self.sampler.rel_tol / 2)
                
        with open(os.path.join(self.sampler.in_path, self.valid_file)) as f:
            f.readline()
            for line in f:
                h, t, r = line.strip().split()
                self.valid_triples.append(
                    (int(t), int(r) + tol, int(h))
                )
                
        with open(os.path.join(self.sampler.in_path, self.test_file)) as f:
            f.readline()
            for line in f:
                h, t, r = line.strip().split()
                self.test_triples.append(
                    (int(t), int(r) + tol, int(h))
                )
                
        self.all_true_triples = set(
            self.triples + self.valid_triples + self.test_triples
        )

    @override
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