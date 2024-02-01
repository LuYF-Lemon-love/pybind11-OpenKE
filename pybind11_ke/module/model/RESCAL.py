# coding:utf-8
#
# pybind11_ke/module/model/RESCAL.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 31, 2023
# 
# 该头文件定义了 RESCAL.

"""
RESCAL - 一个张量分解模型。
"""

import torch
import typing
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class RESCAL(Model):

	"""
	``RESCAL`` :cite:`RESCAL` 提出于 2011 年，是很多张量分解模型的基石，模型较复杂。

	评分函数为:

	.. math::
	
		-\mathbf{h}^T \mathbf{M}_r \mathbf{t}

	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`RESCAL <rescal>`。

	例子::

		from pybind11_ke.utils import WandbLogger
		from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
		from pybind11_ke.module.model import RESCAL
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		from pybind11_ke.config import Trainer, Tester
		
		wandb_logger = WandbLogger(
			project="pybind11-ke",
			name="RESCAL-FB15K237",
			config=dict(
				in_path = '../../benchmarks/FB15K237/',
				batch_size = 2048,
				neg_ent = 25,
				test = True,
				test_batch_size = 10,
				num_workers = 16,
				dim = 50,
				margin = 1.0,
				use_gpu = True,
				device = 'cuda:0',
				epochs = 1000,
				lr = 0.1,
				opt_method = 'adagrad',
				valid_interval = 100,
				log_interval = 100,
				save_interval = 100,
				save_path = '../../checkpoint/rescal.pth'
			)
		)
		
		config = wandb_logger.config
		
		# dataloader for training
		dataloader = KGEDataLoader(
			in_path = config.in_path, 
			batch_size = config.batch_size,
			neg_ent = config.neg_ent,
			test = config.test,
			test_batch_size = config.test_batch_size,
			num_workers = config.num_workers,
			train_sampler = BernSampler,
			test_sampler = TradTestSampler
		)
		
		# define the model
		rescal = RESCAL(
			ent_tol = dataloader.get_ent_tol(),
			rel_tol = dataloader.get_rel_tol(),
			dim = config.dim
		)
		
		# define the loss function
		model = NegativeSampling(
			model = rescal, 
			loss = MarginLoss(margin = config.margin)
		)
		
		# test the model
		tester = Tester(model = rescal, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)
		
		# train the model
		trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(), epochs = config.epochs,
			lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
			tester = tester, test = config.test, valid_interval = config.valid_interval,
			log_interval = config.log_interval, save_interval = config.save_interval,
			save_path = config.save_path, use_wandb = True)
		trainer.run()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim: int = 100):

		"""创建 RESCAL 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		"""

		super(RESCAL, self).__init__(ent_tol, rel_tol)

		#: 实体和关系嵌入向量的维度
		self.dim: int = dim
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
		#: 根据关系个数，创建的关系矩阵
		self.rel_matrices: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim * self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)

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

		head_emb, tail_emb = self.tri2emb(triples, negs, mode)
		rel_matric = self.rel_matrices(triples[:, 1])
		score = self._calc(head_emb, rel_matric, tail_emb)
		return score

	@override
	def tri2emb(
		self,
		triples: torch.Tensor,
		negs: torch.Tensor = None,
		mode: str = 'single') -> tuple[torch.Tensor, torch.Tensor]:

		"""
		返回三元组对应的嵌入向量。
		
		:param triples: 正确的三元组
		:type triples: torch.Tensor
		:param negs: 负三元组类别
		:type negs: torch.Tensor
		:param mode: 模式
		:type triples: str
		:returns: 头实体和尾实体的嵌入向量
		:rtype: tuple[torch.Tensor, torch.Tensor]
		"""
		
		if mode == "single":
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)
			
		elif mode == "head-batch" or mode == "head_predict":
			if negs is None:
				head_emb = self.ent_embeddings.weight.data.unsqueeze(0)
			else:
				head_emb = self.ent_embeddings(negs)
				
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)
			
		elif mode == "tail-batch" or mode == "tail_predict": 
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)
			
			if negs is None:
				tail_emb = self.ent_embeddings.weight.data.unsqueeze(0)
			else:
				tail_emb = self.ent_embeddings(negs)
		
		return head_emb, tail_emb

	def _calc(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor) -> torch.Tensor:

		"""计算 RESCAL 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param r: 关系矩阵。
		:type r: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""
		
		r = r.view(-1, self.dim, self.dim)
		r = r.unsqueeze(dim=1)
		h = h.unsqueeze(dim=-2)
		hr = torch.matmul(h, r)
		hr = hr.squeeze(dim=-2)
		return -torch.sum(hr * t, -1)

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]],
		mode) -> torch.Tensor:
		
		"""RESCAL 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		triples = data["positive_sample"]
		head_emb, tail_emb = self.tri2emb(triples, mode=mode)
		rel_matric = self.rel_matrices(triples[:, 1])
		score = self._calc(head_emb, rel_matric, tail_emb)
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
		pos_head_emb, pos_tail_emb = self.tri2emb(pos_sample)
		pos_rel_transfer = self.rel_matrices(pos_sample[:, 1])
		if mode == "bern":
			neg_head_emb, neg_tail_emb = self.tri2emb(neg_sample)
		else:
			neg_head_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)
		neg_rel_transfer = self.rel_matrices(pos_sample[:, 1])

		pos_regul = (torch.mean(pos_head_emb ** 2) +
					 torch.mean(pos_tail_emb ** 2) +
					 torch.mean(pos_rel_transfer ** 2)) / 3

		neg_regul = (torch.mean(neg_head_emb ** 2) +
					 torch.mean(neg_tail_emb ** 2) +
					 torch.mean(neg_rel_transfer ** 2)) / 3

		regul = (pos_regul + neg_regul) / 2
		
		return regul

def get_rescal_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`RESCAL` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'RESCAL'
			},
			'dim': {
				'values': [50, 100, 200]
			}
		}

	:returns: :py:class:`RESCAL` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'RESCAL'
		},
		'dim': {
			'values': [50, 100, 200]
		}
	}
		
	return parameters_dict