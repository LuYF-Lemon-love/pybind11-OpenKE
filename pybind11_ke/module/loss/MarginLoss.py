# coding:utf-8
#
# pybind11_ke/module/loss/MarginLoss.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
#
# 该脚本定义了 margin-based ranking criterion 损失函数.

"""
MarginLoss - 损失函数类，TransE 原论文中应用这种损失函数完成模型学习。
"""

import torch
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class MarginLoss(Loss):

	"""
	``TransE`` :cite:`TransE` 原论文中应用这种损失函数完成模型训练。
	
	.. Note:: :py:meth:`forward` 中的正样本评分函数的得分应小于负样本评分函数的得分。

	例子::

		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			batch_size = train_dataloader.get_batch_size()
		)
	"""

	def __init__(
		self,
		adv_temperature: float | None = None,
		margin: float = 6.0):

		"""创建 MarginLoss 对象。

		:param adv_temperature: RotatE 提出的自我对抗负采样中的温度。
		:type adv_temperature: float
		:param margin: gamma。
		:type margin: float
		"""

		super(MarginLoss, self).__init__()

		#: gamma
		self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			#: RotatE 提出的自我对抗负采样中的温度。
			self.adv_temperature: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			#: 是否启用 RotatE 提出的自我对抗负采样。
			self.adv_flag: bool = True
		else:
			self.adv_flag: bool = False
	
	def get_weights(
		self,
		n_score: torch.Tensor) -> torch.Tensor:

		"""计算 RotatE 提出的自我对抗负采样中的负样本的分布概率。
		
		:param n_score: 负样本评分函数的得分。
		:type n_score: torch.Tensor
		:returns: 自我对抗负采样中的负样本的分布概率
		:rtype: torch.Tensor
		"""	

		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(
		self,
		p_score: torch.Tensor,
		n_score: torch.Tensor) -> torch.Tensor:
		
		"""计算 margin-based ranking criterion 损失函数。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param p_score: 正样本评分函数的得分。
		:type p_score: torch.Tensor
		:param n_score: 负样本评分函数的得分。
		:type n_score: torch.Tensor
		:returns: 损失值
		:rtype: torch.Tensor
		"""

		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score,
						 -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			
	def predict(
		self,
		p_score: torch.Tensor,
		n_score: torch.Tensor) ->np.ndarray:
		
		"""MarginLoss 的推理方法。
		
		:param p_score: 正样本评分函数的得分。
		:type p_score: torch.Tensor
		:param n_score: 负样本评分函数的得分。
		:type n_score: torch.Tensor
		:returns: 损失值
		:rtype: numpy.ndarray
		"""

		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()

def get_margin_loss_hpo_config() -> dict[str, dict[str, Any]]:

	"""返回 :py:class:`MarginLoss` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'loss': {
				'value': 'MarginLoss'
			},
			'adv_temperature': {
				'value': None
			},
			'margin': {
				'values': [1.0, 3.0]
			}
		}
	
	:returns: :py:class:`MarginLoss` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'loss': {
			'value': 'MarginLoss'
		},
		'adv_temperature': {
			'value': None
		},
		'margin': {
			'values': [1.0, 3.0]
		}
	}
		
	return parameters_dict