# coding:utf-8
#
# pybind11_ke/module/strategy/CompGCNSampling.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 20, 2023
#
# 该脚本定义了 CompGCN 模型的训练策略.

"""
CompGCNSampling - 训练策略类，包含损失函数。
"""

import dgl
import torch
import typing
from ..loss import Loss
from ..model import CompGCN
from .Strategy import Strategy

class CompGCNSampling(Strategy):

	"""
	将模型和损失函数封装到一起，方便模型训练，用于 ``CompGCN`` :cite:`CompGCN`。
	"""

	def __init__(
		self,
		model: CompGCN = None,
		loss: Loss = None,
		smoothing: float = 0.1,
		ent_tol: int = None):
		
		"""创建 CompGCNSampling 对象。
		
		:param model: CompGCN 模型
		:type model: :py:class:`pybind11_ke.module.model.CompGCN`
		:param loss: 损失函数。
		:type loss: :py:class:`pybind11_ke.module.loss.Loss`
		:param smoothing: smoothing
		:type smoothing: float
		:param ent_tol: 实体个数
		:type ent_tol: int
		"""

		super(CompGCNSampling, self).__init__()

		#: CompGCN 模型，即 :py:class:`pybind11_ke.module.model.CompGCN`
		self.model: CompGCN = model
		#: 损失函数，即 :py:class:`pybind11_ke.module.loss.Loss`
		self.loss: Loss = loss
		#: smoothing
		self.smoothing: float = smoothing
		#: 实体个数
		self.ent_tol: int = ent_tol

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
		relation = data['relation']
		norm     = data['norm']
		sample   = data["sample"]
		label    = data["label"]
		score = self.model(graph, relation, norm, sample)
		label = (1.0 - self.smoothing) * label + (1.0 / self.ent_tol)
		loss  = self.loss(score,  label)
		return loss