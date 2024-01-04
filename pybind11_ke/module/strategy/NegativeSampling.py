# coding:utf-8
#
# pybind11_ke/module/strategy/NegativeSampling.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
#
# 该脚本定义了 KGE 模型的训练策略.

"""
NegativeSampling - 训练策略类，包含损失函数。
"""

import torch
from typing import Any
from ..loss import Loss
from ..model import Model
from .Strategy import Strategy

class NegativeSampling(Strategy):

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
		loss: Loss = None,
		batch_size: int = 256,
		regul_rate: float = 0.0,
		l3_regul_rate: float = 0.0):
		
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

		super(NegativeSampling, self).__init__()
		#: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
		self.model: Model = model
		#: 损失函数，即 :py:class:`pybind11_ke.module.loss.Loss`
		self.loss: Loss = loss
		#: batch size
		self.batch_size: int = batch_size
		#: 权重衰减系数
		self.regul_rate: float = regul_rate
		#: l3 正则化系数
		self.l3_regul_rate: float = l3_regul_rate

	def _get_positive_score(self, score: torch.Tensor) -> torch.Tensor:

		"""
		获得正样本的得分，由于底层 C++ 处理模块的原因，
		所以正样本的得分处于前 batch size 位置。

		:param score: 所有样本的得分。
		:type n_score: torch.Tensor
		:returns: 正样本的得分
		:rtype: torch.Tensor
		"""

		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score: torch.Tensor) -> torch.Tensor:

		"""
		获得负样本的得分，由于底层 C++ 处理模块的原因，
		所以正样本的得分处于前 batch size 位置，负样本处于正样本后面。

		:param score: 所有样本的得分。
		:type n_score: torch.Tensor
		:returns: 负样本的得分
		:rtype: torch.Tensor
		"""
				
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
		
		"""计算最后的损失值。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据
		:type data: dict
		:returns: 损失值
		:rtype: torch.Tensor
		"""

		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res

def get_negative_sampling_hpo_config() -> dict[str, dict[str, Any]]:

	"""返回 :py:class:`NegativeSampling` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'regul_rate': {
				'value': 0.0
			},
			'l3_regul_rate': {
				'value': 0.0
			}
		}
	
	:returns: :py:class:`NegativeSampling` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'regul_rate': {
			'value': 0.0
		},
		'l3_regul_rate': {
			'value': 0.0
		}
	}
		
	return parameters_dict