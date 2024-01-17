# coding:utf-8
#
# pybind11_ke/config/RGCNTester.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 3, 2023
#
# 该脚本定义了验证模型类.

"""
Tester - 验证模型类，内部使用 ``tqmn`` 实现进度条。
"""

import base
import torch
import typing
import numpy as np
from tqdm import tqdm
from ..data import TestDataLoader
from ..module.model import Model
from .link_prediction import link_predict

class RGCNTester(object):

    """
    主要用于 KGE 模型的评估。

    例子::

        from pybind11_ke.config import Trainer, Tester

        # test the model
        transe.load_checkpoint('../checkpoint/transe.ckpt')
        tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
        tester.run_link_prediction()
    """

    def __init__(
        self,
        model: Model | None = None,
        data_loader: TestDataLoader | None = None,
        sampling_mode: str = 'link_test',
        use_gpu: bool = True,
        device: str = "cuda:0"):

        """创建 Tester 对象。
        
        :param model: KGE 模型
        :type model: :py:class:`pybind11_ke.module.model.Model`
        :param data_loader: TestDataLoader
        :type data_loader: :py:class:`pybind11_ke.data.TestDataLoader`
        :param sampling_mode: :py:class:`pybind11_ke.data.TestDataLoader` 负采样的方式：``link_test`` or ``link_valid``
        :type sampling_mode: str
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :param device: 使用哪个 gpu
        :type device: str
        """

        #: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
        self.model: Model | None = model
        #: :py:class:`pybind11_ke.data.TestDataLoader`
        self.data_loader: TestDataLoader | None = data_loader
        #: :py:class:`pybind11_ke.data.TestDataLoader` 负采样的方式：``link_test`` or ``link_valid``
        self.sampling_mode: str = sampling_mode
        #: 是否使用 gpu
        self.use_gpu: bool = use_gpu
        #: gpu，利用 ``device`` 构造的 :py:class:`torch.torch.device` 对象
        self.device: torch.torch.device = torch.device(device)
        
        if self.use_gpu:
            self.model.cuda(device = self.device)

        self.val_dataloader = self.data_loader.val_dataloader()
        self.test_dataloader = self.data_loader.test_dataloader()

    def to_var(
        self,
        x: np.ndarray,
        use_gpu: bool) -> torch.Tensor:

        """根据 ``use_gpu`` 返回 ``x`` 的张量
        
        :param x: 数据
        :type x: numpy.ndarray
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :returns: 张量
        :rtype: torch.Tensor
        """

        if use_gpu:
            return x.to(self.device)
        else:
            return x

    def run_link_prediction(self) -> tuple[float, ...]:
        
        """进行链接预测。

        :returns: 经典指标分别为 MR，MRR，Hits@1，Hits@3，Hits@10
        :rtype: tuple[float, ...]
        """

        if self.sampling_mode == "link_valid":
            training_range = tqdm(self.val_dataloader)
        elif self.sampling_mode == "link_test":
            training_range = tqdm(self.test_dataloader)
        self.model.eval()
        results = dict(
            count = 0,
            mr = 0.0,
            mrr = 0.0,
            hit1 = 0.0,
            hit3 = 0.0,
            hit10 = 0.0
        )
        with torch.no_grad():
            for data in training_range:
                ranks = link_predict({
                    "positive_sample": self.to_var(data["positive_sample"], self.use_gpu),
                    "head_label": self.to_var(data["head_label"], self.use_gpu),
                    "tail_label": self.to_var(data["tail_label"], self.use_gpu),
                    "graph": self.to_var(data["graph"], self.use_gpu),
                    "rela": self.to_var(data["rela"], self.use_gpu),
                    "norm": self.to_var(data["norm"], self.use_gpu),
                    "entity": self.to_var(data["entity"], self.use_gpu)
                }, self.model, prediction='all')
                results["count"] += torch.numel(ranks)
                results["mr"] += torch.sum(ranks).item()
                results["mrr"] += torch.sum(1.0 / ranks).item()
                for k in [1, 3, 10]:
                    results['hit{}'.format(k)] += torch.numel(ranks[ranks <= k])

        count = results["count"]
        mr = np.around(results["mr"] / count, decimals=3).item()
        mrr = np.around(results["mrr"] / count, decimals=3).item()
        hit1 = np.around(results["hit1"] / count, decimals=3).item()
        hit3 = np.around(results["hit3"] / count, decimals=3).item()
        hit10 = np.around(results["hit10"] / count, decimals=3).item()
        
        return mr, mrr, hit1, hit3, hit10
    
    def set_sampling_mode(self, sampling_mode: str):
        
        """设置 :py:attr:`sampling_mode`
        
        :param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
        :type sampling_mode: str
        """
        
        self.sampling_mode = sampling_mode