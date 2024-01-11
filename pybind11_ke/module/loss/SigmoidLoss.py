# coding:utf-8
#
# pybind11_ke/module/loss/SigmoidLoss.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 11, 2023
#
# 该脚本定义了 regularized logistic loss 损失函数.

"""
SigmoidLoss - 损失函数类，RotatE 原论文应用这种损失函数完成模型学习。
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class SigmoidLoss(Loss):
	
	"""
	``RotatE`` :cite:`RotatE` 原论文中应用这种损失函数完成模型训练。
	
	.. Note:: :py:meth:`forward` 中的正样本评分函数的得分应大于负样本评分函数的得分。

	例子::

		from pybind11_ke.module.loss import SigmoidLoss
		from pybind11_ke.module.strategy import NegativeSampling

		# define the loss function
		model = NegativeSampling(
			model = rotate, 
			loss = SigmoidLoss(adv_temperature = 2),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 0.0
		)
	"""

	def __init__(
		self,
		adv_temperature: float | None = None):

		"""创建 SigmoidLoss 对象。

		:param adv_temperature: RotatE 提出的自我对抗负采样中的温度。
		:type adv_temperature: float
		"""

		super(SigmoidLoss, self).__init__()
		#: 逻辑函数，类型为 :py:class:`torch.nn.LogSigmoid`。
		self.criterion: torch.nn.LogSigmoid = nn.LogSigmoid()
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

		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(
		self,
		p_score: torch.Tensor,
		n_score: torch.Tensor) -> torch.Tensor:

		"""计算 SigmoidLoss 损失函数。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param p_score: 正样本评分函数的得分。
		:type p_score: torch.Tensor
		:param n_score: 负样本评分函数的得分。
		:type n_score: torch.Tensor
		:returns: 损失值
		:rtype: torch.Tensor
		"""

		if self.adv_flag:
			return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
		else:
			return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

	def predict(
		self,
		p_score: torch.Tensor,
		n_score: torch.Tensor) -> np.ndarray:

		"""SigmoidLoss 的推理方法。
		
		:param p_score: 正样本评分函数的得分。
		:type p_score: torch.Tensor
		:param n_score: 负样本评分函数的得分。
		:type n_score: torch.Tensor
		:returns: 损失值
		:rtype: numpy.ndarray
		"""

		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()