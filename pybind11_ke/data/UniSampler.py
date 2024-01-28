# coding:utf-8
#
# pybind11_ke/data/GraphSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 21, 2024
#
# R-GCN 的数据采样器.

"""
GraphSampler - R-GCN 的数据采样器。
"""

import dgl
import torch
import typing
import warnings
import numpy as np
from .BaseSampler import BaseSampler

warnings.filterwarnings("ignore")

class UniSampler(BaseSampler):

    """``R-GCN`` :cite:`R-GCN` 的训练数据采样器。

    例子::

        from pybind11_ke.data import GraphSampler, CompGCNSampler
        from torch.utils.data import DataLoader

        #: 训练数据采样器
        train_sampler: typing.Union[typing.Type[GraphSampler], typing.Type[CompGCNSampler]] = train_sampler(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            batch_size=batch_size,
            neg_ent=neg_ent
        )

        #: 训练集三元组
        data_train: list[tuple[int, int, int]] = train_sampler.get_train()

        train_dataloader = DataLoader(
            data_train,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_sampler.sampling,
        )
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        neg_ent: int = 1):

        """创建 GraphSampler 对象。

        :param in_path: 数据集目录
        :type in_path: str
        :param ent_file: entity2id.txt
        :type ent_file: str
        :param rel_file: relation2id.txt
        :type rel_file: str
        :param train_file: train2id.txt
        :type train_file: str
        :param batch_size: batch size
        :type batch_size: int | None
        :param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
        :type neg_ent: int
        """

        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file
        )

        #: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
        self.neg_ent: int = neg_ent

        self.cross_sampling_flag = 0

    def sampling(self, data):
        
        """Filtering out positive samples and selecting some samples randomly as negative samples.
        
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        
        batch_data = {}
        neg_ent_sample = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.neg_ent)
                neg_ent_sample.append(neg_head)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.neg_ent)
                neg_ent_sample.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        return batch_data