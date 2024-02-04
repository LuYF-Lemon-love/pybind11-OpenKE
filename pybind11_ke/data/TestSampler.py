# coding:utf-8
#
# pybind11_ke/data/TestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# 测试数据采样器基类.

"""
TestSampler - 测试数据采样器基类。
"""

import os
import torch
import typing
from .TradSampler import TradSampler
from .RGCNSampler import RGCNSampler
from .CompGCNSampler import CompGCNSampler
from collections import defaultdict as ddict
from ..utils import construct_type_constrain

class TestSampler(object):

    """测试数据采样器基类。
    """

    def __init__(
        self,
        sampler: typing.Union[TradSampler, RGCNSampler, CompGCNSampler],
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt",
        type_constrain: bool = True):

        """创建 TestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: typing.Union[TradSampler, RGCNSampler, CompGCNSampler]
        :param valid_file: valid2id.txt
        :type valid_file: str
        :param test_file: test2id.txt
        :type test_file: str
        :param type_constrain: 是否报告 type_constrain.txt 限制的测试结果
        :type type_constrain: bool
        """

        #: 训练数据采样器
        self.sampler: typing.Union[TradSampler, RGCNSampler, CompGCNSampler] = sampler
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

        #: 是否报告 type_constrain.txt 限制的测试结果
        self.type_constrain: bool = type_constrain

        if self.type_constrain:
            construct_type_constrain(
                in_path = self.sampler.in_path,
                train_file = self.sampler.train_file,
                valid_file = self.valid_file,
                test_file = self.test_file
            )
            #: 知识图谱中所有 r 存在头实体种类
            self.rel_heads: ddict[set] = ddict(set)
            #: 知识图谱中所有 r 存在尾实体种类
            self.rel_tails: ddict[set] = ddict(set)
            self.get_type_constrain_id()

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
        
    def get_type_constrain_id(self):

        """读取 type_constrain.txt 文件。"""
                
        with open(os.path.join(self.sampler.in_path, "type_constrain.txt")) as f:
            rel_tol = (int)(f.readline())
            first_line = True
            for line in f:
                rel_types = line.strip().split("\t")
                for entity in rel_types[2:]:
                    if first_line:
                        self.rel_heads[int(rel_types[0])].add(int(entity))
                    else:
                        self.rel_tails[int(rel_types[0])].add(int(entity))
                first_line = not first_line

        for rel in self.rel_heads:
            self.rel_heads[rel] = torch.tensor(list(self.rel_heads[rel]))
        for rel in self.rel_tails:
            self.rel_tails[rel] = torch.tensor(list(self.rel_tails[rel]))

    def get_hr2t_rt2h_from_all(self):

        """获得 :py:attr:`hr2t_all` 和 :py:attr:`rt2h_all` 。"""

        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(
        self,
        data: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:

        """采样函数。
        
        :param data: 测试的正确三元组
        :type data: list[tuple[int, int, int]]
        :returns: 测试数据
        :rtype: dict[str, torch.Tensor]
        """
        
        raise NotImplementedError

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