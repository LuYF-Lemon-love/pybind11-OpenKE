# coding:utf-8
#
# pybind11_ke/module/model/TransE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
# 
# 该头文件定义了 TransE.

"""
TransE - 第一个平移模型，简单而且高效。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from typing_extensions import override

class TransE(Model):

	"""
	``TransE`` :cite:`TransE` 提出于 2013 年，是第一个平移模型，开创了平移模型研究方向。由于其简单性和高效性，
	至今依旧是常用基线模型，在某些数据集上能够比其他更复杂的模型表现的更好。
	
	评分函数为: 
	
	.. math::
	
		\parallel h + r - t \parallel_{L_1/L_2}
	
	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`TransE <transe>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import TransE
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		transe = TransE(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = 50, 
			p_norm = 1, 
			norm_flag = True)
		
		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			batch_size = train_dataloader.get_batch_size()
		)
			
		# test the model
		tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')
		
		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
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
		
		"""创建 TransE 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		:param margin: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
		:type margin: float
		"""
		
		super(TransE, self).__init__(ent_tol, rel_tol)
		
		#: 实体和关系嵌入向量的维度
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

		if margin != None:
			#: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
			self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag: bool = True
		else:
			self.margin_flag: bool = False

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def _calc(
		self,
		h: torch.Tensor,
		t: torch.Tensor,
		r: torch.Tensor) -> torch.Tensor:

		"""计算 TransE 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系的向量。
		:type r: torch.Tensor
		:param mode: ``normal`` 表示 :py:class:`pybind11_ke.data.TrainDataLoader` 
					 为训练同时进行头实体和尾实体负采样的数据，``head_batch`` 和 ``tail_batch`` 
					 表示为了减少数据传输成本，需要进行广播的数据，在广播前需要 reshape。
		:type mode: str
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
	def forward(self, triples, negs=None, mode='single') -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
		score = self._calc(head_emb, tail_emb, relation_emb)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

		"""L2 正则化函数（又称权重衰减），在损失函数中用到。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""
		
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]],
		mode) -> np.ndarray:
		
		"""TransE 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		triples = data["positive_sample"]
		head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
		score = self._calc(head_emb, tail_emb, relation_emb)
		
		if self.margin_flag:
			score = self.margin - score
			return score
		else:
			return -score
			
	def tri2emb(self, triples, negs=None, mode="single"):
		
		if mode == "single":
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
			
		elif mode == "head-batch" or mode == "head_predict":
			if negs is None:  # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
				head_emb = self.ent_embeddings.weight.data.unsqueeze(0)  # [1, num_ent, dim]
			else:
				head_emb = self.ent_embeddings(negs)  # [bs, num_neg, dim]
				
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
			
		elif mode == "tail-batch" or mode == "tail_predict": 
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
			
			if negs is None:
				tail_emb = self.ent_embeddings.weight.data.unsqueeze(0)  # [1, num_ent, dim]
			else:
				tail_emb = self.ent_embeddings(negs)  # [bs, num_neg, dim]
		
		return head_emb, relation_emb, tail_emb

def get_transe_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`TransE` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'TransE'
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

	:returns: :py:class:`TransE` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'TransE'
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