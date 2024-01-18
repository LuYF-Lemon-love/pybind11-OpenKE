# coding:utf-8
#
# pybind11_ke/config/RGCNTester.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 18, 2023
#
# 该脚本定义了 R-GCN 验证模型类.

"""
RGCNTester - R-GCN 验证模型类，内部使用 ``tqmn`` 实现进度条。
"""

import dgl
import torch
import typing
import numpy as np
from tqdm import tqdm
from ..data import GraphDataLoader
from .Tester import Tester
from ..module.model import RGCN
from typing_extensions import override
from torch.utils.data import DataLoader

class RGCNTester(Tester):

    """
    主要用于 ``R-GCN`` :cite:`R-GCN` 模型的评估。
    """

    def __init__(
        self,
        model: RGCN | None = None,
        data_loader: GraphDataLoader | None = None,
        sampling_mode: str = 'link_test',
        use_gpu: bool = True,
        device: str = "cuda:0"):

        """创建 Tester 对象。
        
        :param model: RGCN 模型
        :type model: :py:class:`pybind11_ke.module.model.RGCN`
        :param data_loader: GraphDataLoader
        :type data_loader: :py:class:`pybind11_ke.data.GraphDataLoader`
        :param sampling_mode: 评估验证集还是测试集：``link_test`` or ``link_valid``
        :type sampling_mode: str
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :param device: 使用哪个 gpu
        :type device: str
        """

        super(RGCNTester, self).__init__(
            model=model,
            data_loader=data_loader,
            sampling_mode=sampling_mode,
            use_gpu=use_gpu,
            device=device
        )

        #: 验证数据加载器。
        self.val_dataloader: DataLoader = self.data_loader.val_dataloader()
        #: 测试数据加载器。
        self.test_dataloader: DataLoader = self.data_loader.test_dataloader()

    @override
    def to_var(
        self,
        x: torch.Tensor,
        use_gpu: bool) -> torch.Tensor:

        """根据 ``use_gpu`` 返回 ``x`` 的张量
        
        :param x: 数据
        :type x: torch.Tensor
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :returns: 张量
        :rtype: torch.Tensor
        """

        if use_gpu:
            return x.to(self.device)
        else:
            return x

    @override
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

def link_predict(
    batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]],
    model: RGCN,
    prediction: str = "all") -> torch.Tensor:

    """
    进行链接预测。
    
    :param batch: ``R-GCN`` :cite:`R-GCN` 的测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]
    :param model: ``R-GCN`` :cite:`R-GCN` 模型
    :type model: RGCN
    :param prediction: "all", "head", "tail"
    :type prediction: str
    :returns: 正确三元组的排名
    :rtype: torch.Tensor
    """
    
    if prediction == "all":
        tail_ranks = tail_predict(batch, model)
        head_ranks = head_predict(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict(batch, model)
    elif prediction == "tail":
        ranks = tail_predict(batch, model)

    return ranks.float()

def head_predict(
    batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]],
    model: RGCN) -> torch.Tensor:

    """
    进行头实体的链接预测。
    
    :param batch: ``R-GCN`` :cite:`R-GCN` 的测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]
    :param model: ``R-GCN`` :cite:`R-GCN` 模型
    :type model: RGCN
    :returns: 正确三元组的排名
    :rtype: torch.Tensor
    """
    
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0]
    label = batch["head_label"]
    pred_score = model.predict(batch, "head_predict")
    return calc_ranks(idx, label, pred_score)

def tail_predict(
    batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]],
    model: RGCN) -> torch.Tensor:

    """
    进行尾实体的链接预测。
    
    :param batch: ``R-GCN`` :cite:`R-GCN` 的测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]
    :param model: ``R-GCN`` :cite:`R-GCN` 模型
    :type model: RGCN
    :returns: 正确三元组的排名
    :rtype: torch.Tensor
    """

    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 2]
    label = batch["tail_label"]
    pred_score = model.predict(batch, "tail_predict")
    return calc_ranks(idx, label, pred_score)

def calc_ranks(
    idx: torch.Tensor,
    label: torch.Tensor,
    pred_score: torch.Tensor) -> torch.Tensor:

    """
    计算三元组的排名。
    
    :param idx: 需要链接预测的实体 ID
    :type idx: torch.Tensor
    :param label: 标签
    :type label: torch.Tensor
    :param pred_score: 三元组的评分
    :type pred_score: torch.Tensor
    :returns: 正确三元组的排名
    :rtype: torch.Tensor
    """

    b_range = torch.arange(pred_score.size()[0])
    target_pred = pred_score[b_range, idx]
    pred_score = torch.where(label.bool(), -torch.ones_like(pred_score) * 10000000, pred_score)
    pred_score[b_range, idx] = target_pred

    ranks = (
        1
        + torch.argsort(
            torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
        )[b_range, idx]
    )
    return ranks