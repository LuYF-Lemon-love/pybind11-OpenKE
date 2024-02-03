# coding:utf-8
#
# pybind11_ke/data/TradTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# 平移模型和语义匹配模型的测试数据采样器.

"""
TradTestSampler - 平移模型和语义匹配模型的测试数据采样器。
"""

import os
import torch
from .TradSampler import TradSampler
from .TestSampler import TestSampler
from typing_extensions import override
from collections import defaultdict as ddict
from ..utils import construct_type_constrain

class TradTestSampler(TestSampler):

    """平移模型和语义匹配模型的测试数据采样器。
    """

    def __init__(
        self,
        sampler: TradSampler,
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt",
        type_constrain: bool = True):

        """创建 TradTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: TradSampler
        :param valid_file: valid2id.txt
        :type valid_file: str
        :param test_file: test2id.txt
        :type test_file: str
        :param type_constrain: 是否报告 type_constrain.txt 限制的测试结果
        :type type_constrain: bool
        """

        super().__init__(
            sampler=sampler,
            valid_file=valid_file,
            test_file=test_file
        )

        #: 是否报告 type_constrain.txt 限制的测试结果
        self.type_constrain: bool = type_constrain

        self.get_hr2t_rt2h_from_all()

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

    @override
    def sampling(
        self,
        data: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:

        """采样函数。
        
        :param data: 测试的正确三元组
        :type data: list[tuple[int, int, int]]
        :returns: 测试数据
        :rtype: dict[str, torch.Tensor]
        """
        
        batch_data = {}
        head_label = torch.zeros(len(data), self.ent_tol)
        tail_label = torch.zeros(len(data), self.ent_tol)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0

        if self.type_constrain:
            head_label_type = torch.ones(len(data), self.ent_tol)
            tail_laebl_type = torch.ones(len(data), self.ent_tol)
            for idx, triple in enumerate(data):
                head, rel, tail = triple
                head_label_type[idx][self.rel_heads[rel]] = 0.0
                tail_laebl_type[idx][self.rel_tails[rel]] = 0.0
                head_label_type[idx][self.rt2h_all[(rel, tail)]] = 1.0
                tail_laebl_type[idx][self.hr2t_all[(head, rel)]] = 1.0
            batch_data["head_label_type"] = head_label_type
            batch_data["tail_label_type"] = tail_laebl_type
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data