# coding:utf-8
#
# pybind11_ke/module/strategy/RGCNSampling.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 18, 2023
#
# 该脚本定义了 R-GCN 模型的训练策略.

"""
NegativeSampling - 训练策略类，包含损失函数。
"""

import dgl
import torch
import typing
from ..loss import Loss
from ..model import Model
from .Strategy import Strategy

class RGCNSampling(Strategy):

	"""
	将模型和损失函数封装到一起，方便模型训练，用于 ``R-GCN`` :cite:`R-GCN`。

    例子::

        from pybind11_ke.data import GraphDataLoader
        from pybind11_ke.module.model import RGCN
        from pybind11_ke.module.loss import RGCNLoss
        from pybind11_ke.module.strategy import RGCNSampling
        from pybind11_ke.config import GraphTrainer, GraphTester
        
        dataloader = GraphDataLoader(
        	in_path = "../../benchmarks/FB15K237/",
        	batch_size = 60000,
        	neg_ent = 10,
        	test = True,
        	test_batch_size = 100,
        	num_workers = 16
        )
        
        # define the model
        rgcn = RGCN(
        	ent_tol = dataloader.train_sampler.ent_tol,
        	rel_tol = dataloader.train_sampler.rel_tol,
        	dim = 500,
        	num_layers = 2
        )
        
        # define the loss function
        model = RGCNSampling(
        	model = rgcn,
        	loss = RGCNLoss(model = rgcn, regularization = 1e-5)
        )
        
        # test the model
        tester = GraphTester(model = rgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0')
        
        # train the model
        trainer = GraphTrainer(model = model, data_loader = dataloader.train_dataloader(),
        	epochs = 10000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
        	tester = tester, test = True, valid_interval = 500, log_interval = 500,
        	save_interval = 500, save_path = '../../checkpoint/rgcn.pth'
        )
        trainer.run()
	"""

	def __init__(
		self,
		model: Model = None,
		loss: Loss = None):
		
		"""创建 RGCNSampling 对象。
		
		:param model: R-GCN 模型
		:type model: :py:class:`pybind11_ke.module.model.RGCN`
		:param loss: 损失函数。
		:type loss: :py:class:`pybind11_ke.module.loss.Loss`
		"""

		super(RGCNSampling, self).__init__()

		#: R-GCN 模型，即 :py:class:`pybind11_ke.module.model.RGCN`
		self.model: Model = model
		#: 损失函数，即 :py:class:`pybind11_ke.module.loss.Loss`
		self.loss: Loss = loss

	def forward(
		self,
		data: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]) -> torch.Tensor:
		
		"""计算最后的损失值。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据
		:type data: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
		:returns: 损失值
		:rtype: torch.Tensor
		"""
		
		graph    = data["graph"]
		entity   = data['entity']
		relation = data['relation']
		norm     = data['norm']
		triples  = data["triples"]
		label    = data["label"]
		score = self.model(graph, entity, relation, norm, triples)
		loss  = self.loss(score,  label)
		return loss

def get_rgcn_sampling_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`RGCNSampling` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'strategy': {
				'value': 'RGCNSampling'
			}
		}
	
	:returns: :py:class:`RGCNSampling` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'strategy': {
			'value': 'RGCNSampling'
		}
	}
		
	return parameters_dict