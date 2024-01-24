# coding:utf-8
#
# pybind11_ke/config/GraphTester.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 18, 2023
#
# 该脚本定义了 R-GCN 验证模型类.

"""
GraphTester - R-GCN 验证模型类，内部使用 ``tqmn`` 实现进度条。
"""

import dgl
import torch
import typing
import numpy as np
from tqdm import tqdm
from ..data import GraphDataLoader
from .Tester import Tester
from ..module.model import RGCN, CompGCN
from typing_extensions import override

class GraphTester(Tester):

    """
    主要用于 ``R-GCN`` :cite:`R-GCN` 模型的评估。

	例子::

		from pybind11_ke.data import CompGCNSampler, CompGCNTestSampler, GraphDataLoader
		from pybind11_ke.module.model import CompGCN
		from pybind11_ke.module.loss import Cross_Entropy_Loss
		from pybind11_ke.module.strategy import CompGCNSampling
		from pybind11_ke.config import GraphTrainer, GraphTester
		
		dataloader = GraphDataLoader(
			in_path = "../../benchmarks/FB15K237/",
			batch_size = 2048,
			test_batch_size = 256,
			num_workers = 16,
			train_sampler = CompGCNSampler,
			test_sampler = CompGCNTestSampler
		)
		
		# define the model
		compgcn = CompGCN(
			ent_tol = dataloader.train_sampler.ent_tol,
			rel_tol = dataloader.train_sampler.rel_tol,
			dim = 100
		)
		
		# define the loss function
		model = CompGCNSampling(
			model = compgcn,
			loss = Cross_Entropy_Loss(model = compgcn),
			ent_tol = dataloader.train_sampler.ent_tol
		)
		
		# test the model
		tester = GraphTester(model = compgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")
		
		# train the model
		trainer = GraphTrainer(model = model, data_loader = dataloader.train_dataloader(),
			epochs = 2000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
			tester = tester, test = True, valid_interval = 50, log_interval = 50,
			save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
		)
		trainer.run()
    """

    def __init__(
        self,
        model: RGCN | CompGCN | None = None,
        data_loader: GraphDataLoader | None = None,
        sampling_mode: str = 'link_test',
        prediction: str = "all",
        use_gpu: bool = True,
        device: str = "cuda:0"):

        """创建 Tester 对象。
        
        :param model: RGCN or CompGCN
        :type model: :py:class:`pybind11_ke.module.model.RGCN` or :py:class:`pybind11_ke.module.model.CompGCN`
        :param data_loader: GraphDataLoader
        :type data_loader: :py:class:`pybind11_ke.data.GraphDataLoader`
        :param sampling_mode: 评估验证集还是测试集：'link_test' or 'link_valid'
        :type sampling_mode: str
        :param prediction: 链接预测模式: 'all'、'head'、'tail'
        :type prediction: str
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :param device: 使用哪个 gpu
        :type device: str
        """

        super(GraphTester, self).__init__(
            model=model,
            data_loader=data_loader,
            sampling_mode=sampling_mode,
            use_gpu=use_gpu,
            device=device
        )

        #: 链接预测模式: 'all'、'head'、'tail'
        self.prediction: str = prediction

        #: 验证数据加载器。
        self.val_dataloader: torch.utils.data.DataLoader = self.data_loader.val_dataloader()
        #: 测试数据加载器。
        self.test_dataloader: torch.utils.data.DataLoader = self.data_loader.test_dataloader()

    @override
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
                data = {key : self.to_var(value) for key, value in data.items()}
                ranks = link_predict(data, self.model, prediction=self.prediction)
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

def get_graph_tester_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`GraphTester` 的默认超参数优化配置。
    
    默认配置为::
    
	    parameters_dict = {
            'tester': {
                'value': 'GraphTester'
            },
            'prediction': {
                'value': 'all'
            },
            'use_gpu': {
                'value': True
            },
            'device': {
                'value': 'cuda:0'
            },
        }

    :returns: :py:class:`GraphTester` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]  
    """

	parameters_dict = {
        'tester': {
            'value': 'GraphTester'
        },
        'prediction': {
            'value': 'all'
        },
        'use_gpu': {
            'value': True
        },
        'device': {
            'value': 'cuda:0'
        },
    }
		
	return parameters_dict