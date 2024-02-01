# coding:utf-8
#
# pybind11_ke/module/strategy/NegativeSampling.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2024
#
# 该脚本定义了平移模型和语义匹配模型的训练策略.

"""
NegativeSampling - 训练策略类，包含损失函数。
"""

import torch
import typing
from ..loss import Loss
from ..model import Model
from .Strategy import Strategy

class NegativeSampling(Strategy):

	"""
	将模型和损失函数封装到一起，方便模型训练。

	例子::

		from pybind11_ke.module.model import TransE
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		transe = TransE(
			ent_tol = dataloader.train_sampler.ent_tol,
			rel_tol = dataloader.train_sampler.rel_tol,
			dim = 50, 
			p_norm = 1, 
			norm_flag = True
		)
		
		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			regul_rate = 0.01
		)
	"""

	def __init__(
		self,
		model: Model = None,
		loss: Loss = None,
		regul_rate: float = 0.0,
		l3_regul_rate: float = 0.0):
		
		"""创建 NegativeSampling 对象。

		:param model: KGE 模型
		:type model: :py:class:`pybind11_ke.module.model.Model`
		:param loss: 损失函数。
		:type loss: :py:class:`pybind11_ke.module.loss.Loss`
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
		#: 权重衰减系数
		self.regul_rate: float = regul_rate
		#: l3 正则化系数
		self.l3_regul_rate: float = l3_regul_rate

	def forward(self, data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:
		
		"""计算最后的损失值。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据
		:type data: dict[str, typing.Union[torch.Tensor, str]]
		:returns: 损失值
		:rtype: torch.Tensor
		"""
		
		pos_sample = data["positive_sample"]
		neg_sample = data["negative_sample"]
		mode = data["mode"]
		pos_score = self.model(pos_sample)
		if mode == "bern":
			neg_score = self.model(neg_sample)
			neg_score = neg_score.view(pos_score.shape[0], -1)
		else:
			neg_score = self.model(pos_sample, neg_sample, mode)
		loss_res = self.loss(pos_score, neg_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res

def get_negative_sampling_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`NegativeSampling` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'strategy': {
				'value': 'NegativeSampling'
			},
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
		'strategy': {
			'value': 'NegativeSampling'
		},
		'regul_rate': {
			'value': 0.0
		},
		'l3_regul_rate': {
			'value': 0.0
		}
	}
		
	return parameters_dict