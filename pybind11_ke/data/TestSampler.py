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

import os
import dgl
import torch
import typing
import numpy as np
from .GraphSampler import GraphSampler
from .CompGCNSampler import CompGCNSampler
from collections import defaultdict as ddict

class TestSampler(object):

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

        #: 训练数据采样器
        self.sampler: typing.Union[GraphSampler, CompGCNSampler] = sampler
        #: 实体的个数
        self.ent_tol: int = sampler.ent_tol
        #: valid2id.txt
        self.valid_file: str = valid_file
        #: test2id.txt
        self.test_file: str = test_file

        #: 验证集三元组的个数
        self.valid_tol: int = 0
        #: 测试集三元组的个数
        self.test_tol: int = 0

        #: 验证集三元组
        self.valid_triples: list[tuple[int, int, int]] = []
        #: 测试集三元组
        self.test_triples: list[tuple[int, int, int]] = []
        #: 知识图谱所有三元组
        self.all_true_triples: set[tuple[int, int, int]] = set()

        self.get_valid_test_triples_id()

        #: 知识图谱中所有 h-r 对对应的 t 集合
        self.hr2t_all: ddict[set] = ddict(set)
        #: 知识图谱中所有 r-t 对对应的 h 集合
        self.rt2h_all: ddict[set] = ddict(set)

        self.get_hr2t_rt2h_from_all()

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

        self.all_true_triples = set(
            self.sampler.train_triples + self.valid_triples + self.test_triples
        )

    def get_hr2t_rt2h_from_all(self):

        """获得 :py:attr:`hr2t_all` 和 :py:attr:`rt2h_all` 。"""

        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):

        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
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
        return batch_data

    def get_valid(self) -> list[tuple[int, int, int]]:

        """
        返回验证集三元组。

        :returns: :py:attr:`valid_triples`
        :rtype: list[tuple[int, int, int]]
        """

        return self.valid_triples

    def get_test(self) -> list[tuple[int, int, int]]:

        """
        返回测试集三元组。

        :returns: :py:attr:`test_triples`
        :rtype: list[tuple[int, int, int]]
        """

        return self.test_triples

    def get_all_true_triples(self) -> set[tuple[int, int, int]]:

        """
        返回知识图谱所有三元组。

        :returns: :py:attr:`all_true_triples`
        :rtype: set[tuple[int, int, int]]
        """

        return self.all_true_triples 