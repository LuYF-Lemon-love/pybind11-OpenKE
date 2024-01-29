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

import torch
from .TradSampler import TradSampler
from .TestSampler import TestSampler
from typing_extensions import override

class TradTestSampler(TestSampler):

    """平移模型和语义匹配模型的测试数据采样器。
    """

    def __init__(
        self,
        sampler: TradSampler,
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt"):

        """创建 TradTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: TradSampler
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

        self.get_hr2t_rt2h_from_all()

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
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data