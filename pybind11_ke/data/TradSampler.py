# coding:utf-8
#
# pybind11_ke/data/TradSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 28, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# 为 KGReader 增加构建负三元组的函数，用于平移模型和语义匹配模型.

"""
TradSampler - 为 KGReader 增加构建负三元组的函数，用于平移模型和语义匹配模型。
"""

import torch
import typing
import numpy as np
from .KGReader import KGReader

class TradSampler(KGReader):
    
    """平移模型和语义匹配模型的采样器的基类。
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        batch_size: int | None = None,
        neg_ent: int = 1):

        """创建 TradSampler 对象。

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
            train_file=train_file
        )

        #: batch size
        self.batch_size: int = batch_size
        #: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
        self.neg_ent: int = neg_ent

        self.get_hr2t_rt2h_from_train()

    def sampling(
        self,
        pos_triples: list[tuple[int, int, int]]) -> dict[str, typing.Union[str, torch.Tensor]]:
        
        """平移模型和语义匹配模型的训练集普通的数据采样函数。
        
        :param pos_triples: 知识图谱中的正确三元组
        :type pos_triples: list[tuple[int, int, int]]
        :returns: 平移模型和语义匹配模型的训练数据
        :rtype: dict[str, typing.Union[str, torch.Tensor]]
        """
        
        raise NotImplementedError

    def head_batch(
        self,
        t: int,
        r: int,
        neg_size: int= None) -> np.ndarray:

        """替换头实体构建负三元组。

        :param t: 尾实体
        :type t: int
        :param r: 关系
        :type r: int
        :param neg_size: 负三元组个数
        :type neg_size: int
        :returns: 负三元组中的头实体列表
        :rtype: numpy.ndarray
        """
        
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __corrupt_head(
        self,
        t: int,
        r: int,
        num_max: int = 1) -> np.ndarray:

        """替换头实体构建负三元组。

        :param t: 尾实体
        :type t: int
        :param r: 关系
        :type r: int
        :param num_max: 一次负采样的个数
        :type num_max: int
        :returns: 负三元组的头实体列表
        :rtype: numpy.ndarray
        """
        
        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def tail_batch(
        self,
        h: int,
        r: int,
        neg_size: int = None) -> np.ndarray:
        
        """替换尾实体构建负三元组。

        :param h: 头实体
        :type h: int
        :param r: 关系
        :type r: int
        :param neg_size: 负三元组个数
        :type neg_size: int
        :returns: 负三元组中的尾实体列表
        :rtype: numpy.ndarray
        """
        
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]
        
    def __corrupt_tail(
        self,
        h: int,
        r: int,
        num_max: int = 1) -> np.ndarray:
        
        """替换尾实体构建负三元组。

        :param h: 头实体
        :type h: int
        :param r: 关系
        :type r: int
        :param num_max: 一次负采样的个数
        :type num_max: int
        :returns: 负三元组的尾实体列表
        :rtype: numpy.ndarray
        """
        
        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg