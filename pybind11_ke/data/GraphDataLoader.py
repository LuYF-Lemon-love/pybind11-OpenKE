# coding:utf-8
#
# pybind11_ke/data/GraphDataLoader.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
#
# 读取数据.

"""
GraphDataLoader - 读取数据集类。
"""

import os
import dgl
import torch
import numpy as np
from .GraphSampler import GraphSampler
from .GraphTestSampler import GraphTestSampler
from torch.utils.data import DataLoader

class GraphDataLoader:

    """基本图神经网络采样器。
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt",
        batch_size: int | None = None,
        neg_ent: int = 1,
        test_batch_size: int | None = None,
        num_workers: int | None = None):

        self.in_path = in_path
        self.ent_file = ent_file
        self.rel_file = rel_file
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.batch_size: int = batch_size
        self.neg_ent: int = neg_ent
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_sampler = GraphSampler(
            in_path=self.in_path,
            ent_file=self.ent_file,
            rel_file=self.rel_file,
            train_file=self.train_file,
            valid_file=self.valid_file,
            test_file=self.test_file,
            batch_size=self.batch_size,
            neg_ent=self.neg_ent
        )
        self.test_sampler = GraphTestSampler(
            sampler=self.train_sampler
        )

        self.setup()

    def setup(self):

        self.data_train = self.train_sampler.get_train()
        self.data_val   = self.train_sampler.get_valid()
        self.data_test  = self.train_sampler.get_test()

    def train_dataloader(self):

        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_sampler.sampling,
        )
            
    def val_dataloader(self):

        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_sampler.sampling,
        )

    def test_dataloader(self):
        
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_sampler.sampling,
        )