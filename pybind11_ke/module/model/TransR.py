# coding:utf-8
#
# pybind11_ke/module/model/TransR.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 21, 2024
# 
# 该头文件定义了 TransR.

"""
TransR - 是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from typing_extensions import override

class TransR(Model):

	"""
	``TransR`` :cite:`TransR` 提出于 2015 年，是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。
	
	评分函数为:
	
	.. math::

		\Vert hM_r+r-tM_r \Vert_{L_1/L_2}

	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`TransR <transr>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import TransE, TransR
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		from pybind11_ke.data import TrainDataLoader, TestDataLoader

		# dataloader for training
		train_dataloader = TrainDataLoader(
			in_path = "../../benchmarks/FB15K237/", 
			nbatches = 100,
			threads = 8, 
			sampling_mode = "normal", 
			bern = True, 
			neg_ent = 25,
			neg_rel = 0)

		# define the transe
		transe = TransE(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = 100, 
			p_norm = 1, 
			norm_flag = True)

		transr = TransR(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim_e = 100,
			dim_r = 100,
			p_norm = 1, 
			norm_flag = True,
			rand_init = False)

		model_e = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 5.0),
			batch_size = train_dataloader.get_batch_size()
		)

		model_r = NegativeSampling(
			model = transr,
			loss = MarginLoss(margin = 4.0),
			batch_size = train_dataloader.get_batch_size()
		)

		# pretrain transe
		trainer = Trainer(model = model_e, data_loader = train_dataloader,
			epochs = 1, lr = 0.5, use_gpu = True, device = 'cuda:1')
		trainer.run()
		parameters = transe.get_parameters()
		transe.save_parameters("../../checkpoint/transr_transe.json")

		# dataloader for test
		test_dataloader = TestDataLoader("../../benchmarks/FB15K237/")

		# test the transr
		tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

		# train transr
		transr.set_parameters(parameters)
		trainer = Trainer(model = model_r, data_loader = train_dataloader,
			epochs = 1000, lr = 1.0, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transr.pth')
		trainer.run()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim_e: int = 100,
		dim_r: int = 100,
		p_norm: int = 1,
		norm_flag: bool = True,
		rand_init: bool = False,
		margin: float | None = None):

		"""创建 TransR 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim_e: 实体嵌入向量的维度
		:type dim_e: int
		:param dim_r: 关系嵌入向量的维度
		:type dim_r: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		:param rand_init: 关系矩阵是否采用随机初始化。
		:type rand_init: bool
		:param margin: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
		:type margin: float
		"""

		super(TransR, self).__init__(ent_tol, rel_tol)
		
		#: 实体嵌入向量的维度
		self.dim_e: int = dim_e
		#: 关系嵌入向量的维度
		self.dim_r: int = dim_r
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm: int = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag: bool = norm_flag
		#: 关系矩阵是否采用随机初始化
		self.rand_init: bool = rand_init

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim_e)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim_r)

		if margin != None:
			#: 当使用 ``RotatE`` :cite:`RotatE` 的损失函数 :py:class:`pybind11_ke.module.loss.SigmoidLoss`，需要提供此参数，将 ``TransE`` :cite:`TransE` 的正三元组的评分由越小越好转化为越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。
			self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag: bool = True
		else:
			self.margin_flag: bool = False

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

		#: 关系矩阵
		self.transfer_matrix: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim_e * self.dim_r)

		if not self.rand_init:
			identity = torch.zeros(self.dim_e, self.dim_r)
			for i in range(min(self.dim_e, self.dim_r)):
				identity[i][i] = 1
			identity = identity.view(self.dim_e * self.dim_r)
			for i in range(self.rel_tol):
				self.transfer_matrix.weight.data[i] = identity
		else:
			nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

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
		rel_transfer = self.transfer_matrix(triples[:, 1])
		head_emb = self._transfer(head_emb, rel_transfer)
		tail_emb = self._transfer(tail_emb, rel_transfer)
		score = self._calc(head_emb, relation_emb, tail_emb)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def _transfer(
		self,
		e: torch.Tensor,
		r_transfer: torch.Tensor) -> torch.Tensor:

		"""
		将头实体或尾实体的向量投影到特定的关系向量空间。
		
		:param e: 头实体或尾实体向量。
		:type e: torch.Tensor
		:param r_transfer: 特定关系矩阵
		:type r_transfer: torch.Tensor
		:returns: 投影后的实体向量
		:rtype: torch.Tensor
		"""

		r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
		r_transfer = r_transfer.unsqueeze(dim=1)
		e = e.unsqueeze(dim=-2)
		e = torch.matmul(e, r_transfer)
		return e.squeeze(dim=-2)

	def _calc(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor) -> torch.Tensor:

		"""计算 TransR 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param r: 关系的向量。
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
		mode) -> torch.Tensor:
		
		"""TransR 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		triples = data["positive_sample"]
		head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
		rel_transfer = self.transfer_matrix(triples[:, 1])
		head_emb = self._transfer(head_emb, rel_transfer)
		tail_emb = self._transfer(tail_emb, rel_transfer)
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
		pos_rel_transfer = self.transfer_matrix(pos_sample[:, 1])
		if mode == "bern":
			neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(neg_sample)
		else:
			neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)
		neg_rel_transfer = self.transfer_matrix(pos_sample[:, 1])

		pos_regul = (torch.mean(pos_head_emb ** 2) + 
					 torch.mean(pos_relation_emb ** 2) + 
					 torch.mean(pos_tail_emb ** 2) +
					 torch.mean(pos_rel_transfer ** 2)) / 4

		neg_regul = (torch.mean(neg_head_emb ** 2) + 
					 torch.mean(neg_relation_emb ** 2) + 
					 torch.mean(neg_tail_emb ** 2) +
					 torch.mean(neg_rel_transfer ** 2)) / 4

		regul = (pos_regul + neg_regul) / 2

		return regul

def get_transr_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`TransR` 的默认超参数优化配置。
	
	``TransR`` :cite:`TransR` 进行超参数优化的时候，需要先训练一个 ``TransE`` :cite:`TransE` 模型（训练 1 epoch）。
	然后 ``TransR`` :cite:`TransR` 的实体和关系的嵌入向量初始化为 TransE 的结果。
	``margin_e``、``lr_e`` 和 ``opt_method_e`` 是 ``TransE`` :cite:`TransE` 的训练超参数。
	如果想获得更详细的信息请访问 :ref:`TransR <transr>`。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'TransR'
			},
			'dim': {
				'values': [50, 100, 200]
			},
			'p_norm': {
				'values': [1, 2]
			},
			'norm_flag': {
				'value': True
			},
			'rand_init': {
				'value': False
			},
			'margin_e': {
				'values': [1.0, 3.0, 6.0]
			},
			'lr_e': {
				'distribution': 'uniform',
				'min': 0,
				'max': 1.0
			},
			'opt_method_e': {
				'values': ['adam', 'adagrad', 'sgd']
			},
		}

	:returns: :py:class:`TransR` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'TransR'
		},
		'dim': {
			'values': [50, 100, 200]
		},
		'p_norm': {
			'values': [1, 2]
		},
		'norm_flag': {
			'value': True
		},
		'rand_init': {
			'value': False
		},
		'margin_e': {
			'values': [1.0, 3.0, 6.0]
		},
		'lr_e': {
			'distribution': 'uniform',
			'min': 0,
			'max': 1.0
		},
		'opt_method_e': {
			'values': ['adam', 'adagrad', 'sgd']
		},
	}
		
	return parameters_dict