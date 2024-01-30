# coding:utf-8
#
# pybind11_ke/module/model/TransH.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 30, 2024
# 
# 该头文件定义了 TransH.

"""
TransH - 是第二个平移模型，将关系建模为超平面上的平移操作。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from typing_extensions import override

class TransH(Model):

	"""
	``TransH`` :cite:`TransH` 提出于 2014 年，是第二个平移模型，将关系建模为超平面上的平移操作。
	
	评分函数为:
	
	.. math::

		\Vert (h-r_w^T hr_w)+r_d-(t-r_w^T tr_w)\Vert_{L_1/L_2}

	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`TransH <transh>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import TransH
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		transh = TransH(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = 200, 
			p_norm = 1, 
			norm_flag = True)
		
		# define the loss function
		model = NegativeSampling(
			model = transh, 
			loss = MarginLoss(margin = 4.0),
			batch_size = train_dataloader.get_batch_size()
		)
		
		# test the model
		tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')
		
		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			epochs = 1000, lr = 0.5, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transe.pth')
		trainer.run()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim: int = 100,
		p_norm: int = 1,
		norm_flag: bool = True,
		margin: float | None = None):
		
		"""创建 TransH 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体、关系嵌入向量和和法向量的维度
		:type dim: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		:param margin: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
		:type margin: float
		"""

		super(TransH, self).__init__(ent_tol, rel_tol)
		
		#: 实体、关系嵌入向量和和法向量的维度
		self.dim: int = dim
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm: int = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag: bool = norm_flag
		
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)
		#: 根据关系个数，创建的法向量
		self.norm_vector: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)

		if margin != None:
			#: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
			self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.norm_vector.weight.data)

	@override
	def forward(
		self,
		triples: torch.Tensor,
		negs: torch.Tensor = None,
		mode: str = 'single') -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param triples: 正确的三元组
		:type triples: torch.Tensor
		:param negs: 负三元组类别
		:type negs: torch.Tensor
		:param mode: 模式
		:type triples: str
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
		norm_vector = self.norm_vector(triples[:, 1]).unsqueeze(dim=1)
		head_emb = self._transfer(head_emb, norm_vector)
		tail_emb = self._transfer(tail_emb, norm_vector)
		score = self._calc(head_emb, relation_emb, tail_emb)

		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def _transfer(
		self,
		e: torch.Tensor,
		norm: torch.Tensor) -> torch.Tensor:

		"""
		将头实体或尾实体的向量投影到超平面上。
		
		:param e: 头实体或尾实体向量。
		:type e: torch.Tensor
		:param norm: 法向量
		:type norm: torch.Tensor
		:returns: 投影后的实体向量
		:rtype: torch.Tensor
		"""

		norm = F.normalize(norm, p = 2, dim = -1)
		return e - torch.sum(e * norm, -1, True) * norm

	def _calc(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor) -> torch.Tensor:

		"""计算 TransH 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param r: 关系实体的向量。
		:type r: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		# 对嵌入的最后一维进行归一化
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		
		score = (h + r) - t
		
		# 利用距离函数计算得分
		score = torch.norm(score, self.p_norm, -1)
		return score

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]],
		mode) -> np.ndarray:
		
		"""TransH 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		triples = data["positive_sample"]
		head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
		norm_vector = self.norm_vector(triples[:, 1]).unsqueeze(dim=1)
		head_emb = self._transfer(head_emb, norm_vector)
		tail_emb = self._transfer(tail_emb, norm_vector)
		score = self._calc(head_emb, relation_emb, tail_emb)
		
		if self.margin_flag:
			score = self.margin - score
			return score
		else:
			return -score

	def regularization(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

		"""L2 正则化函数（又称权重衰减），在损失函数中用到。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor, str]]
		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		pos_sample = data["positive_sample"]
		neg_sample = data["negative_sample"]
		mode = data["mode"]
		pos_head_emb, pos_relation_emb, pos_tail_emb = self.tri2emb(pos_sample)
		pos_norm_vector = self.norm_vector(pos_sample[:, 1]).unsqueeze(dim=1)
		neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)
		neg_norm_vector = self.norm_vector(pos_sample[:, 1]).unsqueeze(dim=1)

		pos_regul = (torch.mean(pos_head_emb ** 2) + 
					 torch.mean(pos_relation_emb ** 2) + 
					 torch.mean(pos_tail_emb ** 2) +
					 torch.mean(pos_norm_vector ** 2)) / 4

		neg_regul = (torch.mean(neg_head_emb ** 2) + 
					 torch.mean(neg_relation_emb ** 2) + 
					 torch.mean(neg_tail_emb ** 2) +
					 torch.mean(neg_norm_vector ** 2)) / 4

		regul = (pos_regul + neg_regul) / 2
		
		return regul

def get_transh_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`TransH` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'TransH'
			},
			'dim': {
				'values': [50, 100, 200]
			},
			'p_norm': {
				'values': [1, 2]
			},
			'norm_flag': {
				'value': True
			}
		}

	:returns: :py:class:`TransH` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'TransH'
		},
		'dim': {
			'values': [50, 100, 200]
		},
		'p_norm': {
			'values': [1, 2]
		},
		'norm_flag': {
			'value': True
		}
	}
		
	return parameters_dict
