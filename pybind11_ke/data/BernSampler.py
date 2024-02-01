# coding:utf-8
#
# pybind11_ke/data/BernSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 30, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 30, 2024
#
# 该脚本定义了 BernSampler 类.

"""
BernSampler - 平移模型和语义匹配模型的训练集数据采样器。
"""

import torch
import typing
import random
import collections
import numpy as np
from .TradSampler import TradSampler
from typing_extensions import override

class BernSampler(TradSampler):

    """平移模型和语义匹配模型的训练集 Bern 数据采样器（伯努利分布），如果想获得更详细的信息请访问 :ref:`TransH <transh>`。。
    """

    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        batch_size: int | None = None,
        neg_ent: int = 1):

        """创建 BernSampler 对象。

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
        :param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity
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
        
        self.tph, self.hpt = self.get_tph_hpt()

    def get_tph_hpt(self) -> tuple[collections.defaultdict[float], collections.defaultdict[float]]:

        """计算 tph 和 hpt。
        
        :returns: tph 和 hpt
        :rtype: tuple[collections.defaultdict[float], collections.defaultdict[float]]
        """

        h_of_r = collections.defaultdict(set)
        t_of_r = collections.defaultdict(set)
        freq_rel = collections.defaultdict(float)
        tph = collections.defaultdict(float)
        hpt = collections.defaultdict(float)
        for h, r, t in self.train_triples:
            freq_rel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            tph[r] = freq_rel[r] / len(h_of_r[r])
            hpt[r] = freq_rel[r] / len(t_of_r[r])
        return tph, hpt

    @override
    def sampling(
        self,
        pos_triples: list[tuple[int, int, int]]) -> dict[str, typing.Union[str, torch.Tensor]]:

        """平移模型和语义匹配模型的训练集 bern 的数据采样函数（伯努利分布）。
        
        :param pos_triples: 知识图谱中的正确三元组
        :type pos_triples: list[tuple[int, int, int]]
        :returns: 平移模型和语义匹配模型的训练数据
        :rtype: dict[str, typing.Union[str, torch.Tensor]]
        """

        batch_data = {}
        neg_ent_sample = []

        batch_data['mode'] = 'bern'
        for h, r, t in pos_triples:
            neg_ent = self.__normal_batch(h, r, t, self.neg_ent)
            neg_ent_sample += neg_ent
        
        batch_data["positive_sample"] = torch.LongTensor(np.array(pos_triples))
        batch_data["negative_sample"] = torch.LongTensor(np.array(neg_ent_sample))

        return batch_data

    def __normal_batch(
        self,
        h: int,
        r: int,
        t: int,
        neg_size: int) -> list[tuple[int, int, int]]:

        """Bern 负采样函数
        
        :param h: 头实体
        :type h: int
        :param r: 关系
        :type r: int
        :param t: 尾实体
        :type t: int
        :param neg_size: 负三元组个数
        :type neg_size: int
        :returns: 负三元组中的头实体列表
        :rtype: list[tuple[int, int, int]]
        """

        neg_size_h = 0
        neg_size_t = 0
        prob = self.hpt[r] / (self.hpt[r] + self.tph[r])
        for _ in range(neg_size):
            if random.random() < prob:
                neg_size_t += 1
            else:
                neg_size_h += 1

        res = []

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
        
        for hh in neg_list_h[:neg_size_h]:
            res.append((hh, r, t))
        
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)
        
        for tt in neg_list_t[:neg_size_t]:
            res.append((h, r, tt))

        return res