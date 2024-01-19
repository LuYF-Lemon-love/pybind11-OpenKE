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