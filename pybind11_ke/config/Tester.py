# coding:utf-8
#
# pybind11_ke/config/Tester.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2023
#
# 该脚本定义了验证模型类.

"""
Tester - 验证模型类，内部使用 ``tqmn`` 实现进度条。
"""

import dgl
import torch
import typing
import collections
import numpy as np
from tqdm import tqdm
from ..module.model import Model
from ..data import KGEDataLoader

class Tester(object):

    """
    主要用于 KGE 模型的评估。
    
    例子::
    
        from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
        from pybind11_ke.module.model import TransE
        from pybind11_ke.module.loss import MarginLoss
        from pybind11_ke.module.strategy import NegativeSampling
        from pybind11_ke.config import Trainer, Tester

        # dataloader for training
        dataloader = KGEDataLoader(
        	in_path = "../../benchmarks/FB15K/", 
        	batch_size = 8192,
        	neg_ent = 25,
        	test = True,
        	test_batch_size = 256,
        	num_workers = 16,
        	train_sampler = BernSampler,
        	test_sampler = TradTestSampler
        )

        # define the model
        transe = TransE(
        	ent_tol = dataloader.train_sampler.ent_tol,
        	rel_tol = dataloader.train_sampler.rel_tol,
        	dim = 50, 
        	p_norm = 1, 
        	norm_flag = True)

        # define the loss function
        model = NegativeSampling(
        	model = transe, 
        	loss = MarginLoss(margin = 1.0),
        	regul_rate = 0.01
        )

        # test the model
        tester = Tester(model = transe, data_loader = dataloader, use_gpu = True, device = 'cuda:1')

        # train the model
        trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
        	epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
        	tester = tester, test = True, valid_interval = 100,
        	log_interval = 100, save_interval = 100,
        	save_path = '../../checkpoint/transe.pth', delta = 0.01)
        trainer.run()
    """

    #: 准备报告的指标 Hit@N 的列表，默认为 [1, 3, 10], 表示报告 Hits@1, Hits@3, Hits@10
    hits: list[int] = [1, 3, 10]

    def __init__(
        self,
        model: typing.Union[Model, None] = None,
        data_loader: KGEDataLoader | None = None,
        sampling_mode: str = 'link_test',
        prediction: str = "all",
        use_tqdm: bool = True,
        use_gpu: bool = True,
        device: str = "cuda:0"):

        """创建 Tester 对象。
        
        :param model: KGE 模型
        :type model: pybind11_ke.module.model.Model
        :param data_loader: KGEDataLoader
        :type data_loader: pybind11_ke.data.KGEDataLoader
        :param sampling_mode: 评估验证集还是测试集：'link_test' or 'link_valid'
        :type sampling_mode: str
        :param prediction: 链接预测模式: 'all'、'head'、'tail'
        :type prediction: str
        :param use_tqdm: 是否启用进度条
        :type use_tqdm: bool
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :param device: 使用哪个 gpu
        :type device: str
        """

        #: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
        self.model: typing.Union[Model, None] = model
        #: :py:class:`pybind11_ke.data.TestDataLoader` or :py:class:`pybind11_ke.data.GraphDataLoader`
        self.data_loader: KGEDataLoader | None = data_loader
        #: :py:class:`pybind11_ke.data.TestDataLoader` 负采样的方式：``link_test`` or ``link_valid``
        self.sampling_mode: str = sampling_mode
        #: 链接预测模式: 'all'、'head'、'tail'
        self.prediction: str = prediction
        #: 是否启用进度条
        self.use_tqdm: bool = use_tqdm
        #: 是否使用 gpu
        self.use_gpu: bool = use_gpu
        #: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
        self.device: torch.device = torch.device(device)
        #: 验证数据加载器。
        self.val_dataloader: torch.utils.data.DataLoader = self.data_loader.val_dataloader()
        #: 测试数据加载器。
        self.test_dataloader: torch.utils.data.DataLoader = self.data_loader.test_dataloader()
        
        if self.use_gpu:
            self.model.cuda(device = self.device)

    def set_hits(
        self,
        new_hits: list[int] = [1, 3, 10]):

        """定义 Hits 指标。
        
        :param new_hits: 准备报告的指标 Hit@N 的列表，默认为 [1, 3, 10], 表示报告 Hits@1, Hits@3, Hits@10
        :type new_hits: list[int]
        """

        tmp = self.hits
        self.hits = new_hits

        print(f"Hits@N 指标由 {tmp} 变为 {self.hits}")

    def to_var(
        self,
        x: torch.Tensor) -> torch.Tensor:

        """根据 :py:attr:`use_gpu` 返回 ``x`` 的张量
        
        :param x: 数据
        :type x: torch.Tensor
        :returns: 张量
        :rtype: torch.Tensor
        """

        if self.use_gpu:
            return x.to(self.device)
        else:
            return x

    def run_link_prediction(self) -> dict[str, float]:
        
        """进行链接预测。

        :returns: 经典指标分别为 MR，MRR，Hits@1，Hits@3，Hits@10
        :rtype: dict[str, float]
        """

        if self.sampling_mode == "link_valid":
            training_range = tqdm(self.val_dataloader) if self.use_tqdm else self.val_dataloader
        elif self.sampling_mode == "link_test":
            training_range = tqdm(self.test_dataloader) if self.use_tqdm else self.test_dataloader
        self.model.eval()
        results = collections.defaultdict(float)
        with torch.no_grad():
            for data in training_range:
                data = {key : self.to_var(value) for key, value in data.items()}
                ranks = link_predict(data, self.model, prediction=self.prediction)
                results["count"] += torch.numel(ranks)
                results["mr"] += torch.sum(ranks).item()
                results["mrr"] += torch.sum(1.0 / ranks).item()
                for k in self.hits:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        count = results["count"]
        results = {key : np.around(value / count, decimals=3).item() for key, value in results.items() if key != "count"}
        
        return results
    
    def set_sampling_mode(self, sampling_mode: str):
        
        """设置 :py:attr:`sampling_mode`
        
        :param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
        :type sampling_mode: str
        """
        
        self.sampling_mode = sampling_mode

def link_predict(
    batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]],
    model: Model,
    prediction: str = "all") -> torch.Tensor:

    """
    进行链接预测。
    
    :param batch: 测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
    :param model: KGE 模型
    :type model: pybind11_ke.module.model.Model
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
    batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]],
    model: Model) -> torch.Tensor:

    """
    进行头实体的链接预测。
    
    :param batch: 测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
    :param model: KGE 模型
    :type model: pybind11_ke.module.model.Model
    :returns: 正确三元组的排名
    :rtype: torch.Tensor
    """
    
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0]
    label = batch["head_label"]
    pred_score = model.predict(batch, "head_predict")
    return calc_ranks(idx, label, pred_score)

def tail_predict(
    batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]],
    model: Model) -> torch.Tensor:

    """
    进行尾实体的链接预测。
    
    :param batch: 测试数据
    :type batch: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
    :param model: KGE 模型
    :type model: pybind11_ke.module.model.Model
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

def get_tester_hpo_config() -> dict[str, dict[str, typing.Any]]:
    
    """返回 :py:class:`Tester` 的默认超参数优化配置。
    
    默认配置为::
    
        parameters_dict = {
            'tester': {
                'value': 'Tester'
            },
            'prediction': {
                'value': 'all'
            },
            'use_tqdm': {
                'value': False
            },
            'use_gpu': {
                'value': True
            },
            'device': {
                'value': 'cuda:0'
            },
        }

    :returns: :py:class:`Tester` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]  
    """
    
    parameters_dict = {
        'tester': {
            'value': 'Tester'
        },
        'prediction': {
            'value': 'all'
        },
        'use_tqdm': {
            'value': False
        },
        'use_gpu': {
            'value': True
        },
        'device': {
            'value': 'cuda:0'
        },
    }
    
    return parameters_dict