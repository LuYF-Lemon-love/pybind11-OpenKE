# coding:utf-8
#
# pybind11_ke/module/strategy/RGCNSampling.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
#
# 该脚本定义了 KGE 模型的训练策略.

"""
NegativeSampling - 训练策略类，包含损失函数。
"""

import torch
import typing
from ..loss import Loss
from ..model import Model
from .Strategy import Strategy

class RGCNSampling(Strategy):

	"""
	将模型和损失函数封装到一起，方便模型训练。
	
	例子::

		from pybind11_ke.config import Trainer
		from pybind11_ke.module.model import TransE
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		transe = TransE(
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
			dim = 50, 
			p_norm = 1, 
			norm_flag = True)
		
		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			batch_size = train_dataloader.get_batch_size()
		)
		
		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			train_times = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transe.pth')
		trainer.run()
	"""

	def __init__(
		self,
		model: Model = None,
		loss: Loss = None):
		
		"""创建 NegativeSampling 对象。

		:param model: KGE 模型
		:type model: :py:class:`pybind11_ke.module.model.Model`
		:param loss: 损失函数。
		:type loss: :py:class:`pybind11_ke.module.loss.Loss`
		:param batch_size: batch size
		:type batch_size: int
		:param regul_rate: 权重衰减系数
		:type regul_rate: float
		:param l3_regul_rate: l3 正则化系数
		:type l3_regul_rate: float
		"""

		super(RGCNSampling, self).__init__()
		#: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
		self.model: Model = model
		#: 损失函数，即 :py:class:`pybind11_ke.module.loss.Loss`
		self.loss: Loss = loss

	def forward(self, data: dict[str, typing.Union[torch.Tensor,str]]) -> torch.Tensor:
		
		"""计算最后的损失值。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 损失值
		:rtype: torch.Tensor
		"""
		
		graph    = data["graph"]
		triples  = data["triples"]
		label    = data["label"]
		entity   = data['entity']
		relation = data['relation']
		norm     = data['norm']
		score = self.model(graph, entity, relation, norm, triples)
		loss  = self.loss(score,  label)
		return loss