# coding:utf-8
#
# pybind11_ke/data/UniSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# 平移模型和语义匹配模型的训练集数据采样器.

"""
UniSampler - 平移模型和语义匹配模型的训练集数据采样器。
"""

import torch
import typing
import warnings
import numpy as np
from .TradSampler import TradSampler

warnings.filterwarnings("ignore")

class UniSampler(TradSampler):

    """平移模型和语义匹配模型的训练集普通的数据采样器。
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        batch_size: int | None = None,
        neg_ent: int = 1):

        """创建 UniSampler 对象。

        :param in_path: 数据集目录
        :type in_path: str
        :param ent_file: entity2id.txt
        :type ent_file: str
        :param rel_file: relation2id.txt
        :type rel_file: str
        :param train_file: train2id.txt
        :type train_file: str
        :param batch_size: batch size 在该采样器中不起作用，只是占位符。
        :type batch_size: int | None
        :param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
        :type neg_ent: int
        """

        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            batch_size = batch_size,
            neg_ent = neg_ent
        )

        self.cross_sampling_flag = 0

    def sampling(
        self,
        pos_triples: list[tuple[int, int, int]]) -> dict[str, typing.Union[str, torch.Tensor]]:
        
        """平移模型和语义匹配模型的训练集普通的数据采样函数。
        
        :param pos_triples: 知识图谱中的正确三元组
        :type pos_triples: list[tuple[int, int, int]]
        :returns: 平移模型和语义匹配模型的训练数据
        :rtype: dict[str, typing.Union[str, torch.Tensor]]
        """
        
        batch_data = {}
        neg_ent_sample = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in pos_triples:
                neg_head = self.head_batch(t, r, self.neg_ent)
                neg_ent_sample.append(neg_head)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in pos_triples:
                neg_tail = self.tail_batch(h, r, self.neg_ent)
                neg_ent_sample.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(pos_triples))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        return batch_data