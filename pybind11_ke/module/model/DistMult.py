# coding:utf-8
#
# pybind11_ke/module/model/DistMult.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 6, 2023
# 
# 该头文件定义了 DistMult.

"""
DistMult - 最简单的双线性模型，与 TransE 参数量相同，因此非常容易的应用于大型的知识图谱。
"""

import torch
import typing
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class DistMult(Model):

	"""
	``DistMult`` :cite:`DistMult` 提出于 2015 年，最简单的双线性模型，与 TransE 参数量相同，因此非常容易的应用于大型的知识图谱。

	评分函数为:

	.. math::
	
		\sum_{i=1}^{n}h_ir_it_i
	
	为逐元素多线性点积（element-wise multi-linear dot product），正三元组的评分函数的值越大越好，负三元组越小越好，如果想获得更详细的信息请访问 :ref:`DistMult <distMult>`。

	例子::

		from pybind11_ke.utils import WandbLogger
		from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
		from pybind11_ke.module.model import DistMult
		from pybind11_ke.module.loss import SoftplusLoss
		from pybind11_ke.module.strategy import NegativeSampling
		from pybind11_ke.config import Trainer, Tester
		
		wandb_logger = WandbLogger(
			project="pybind11-ke",
			name="DistMult-WN18RR",
			config=dict(
				in_path = '../../benchmarks/WN18RR/',
				batch_size = 4096,
				neg_ent = 25,
				test = True,
				test_batch_size = 10,
				num_workers = 16,
				dim = 200,
				regul_rate = 1.0,
				use_gpu = True,
				device = 'cuda:0',
				epochs = 2000,
				lr = 0.5,
				opt_method = 'adagrad',
				valid_interval = 100,
				log_interval = 100,
				save_interval = 100,
				save_path = '../../checkpoint/distMult.pth'
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
		distmult = DistMult(
			ent_tol = dataloader.get_ent_tol(),
			rel_tol = dataloader.get_rel_tol(),
			dim = config.dim
		)
		
		# define the loss function
		model = NegativeSampling(
			model = distmult, 
			loss = SoftplusLoss(), 
			regul_rate = config.regul_rate
		)
		
		# test the model
		tester = Tester(model = distmult, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)
		
		# train the model
		trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(), epochs = config.epochs,
			lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
			tester = tester, test = config.test, valid_interval = config.valid_interval,
			log_interval = config.log_interval, save_interval = config.save_interval,
			save_path = config.save_path, use_wandb = True)
		trainer.run()
		
		# close your wandb run
		wandb_logger.finish()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim: int = 100):

		"""创建 DistMult 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体嵌入向量和关系对角矩阵的维度
		:type dim: int
		"""

		super(DistMult, self).__init__(ent_tol, rel_tol)

		#: 实体嵌入向量和关系对角矩阵的维度
		self.dim: int = dim
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
		#: 根据关系个数，创建的关系对角矩阵
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

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
		score = self._calc(head_emb, relation_emb, tail_emb)
		return score

	def _calc(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor) -> torch.Tensor:

		"""计算 DistMult 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param r: 关系的对角矩阵。
		:type r: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		score = (h * r) * t

		# 计算得分
		score = torch.sum(score, -1)
		return score

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]],
		mode) -> torch.Tensor:
		
		"""DistMult 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		triples = data["positive_sample"]
		head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
		score = self._calc(head_emb, relation_emb, tail_emb)
		return score

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
		if mode == "bern":
			neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(neg_sample)
		else:
			neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)

		pos_regul = (torch.mean(pos_head_emb ** 2) + 
					 torch.mean(pos_relation_emb ** 2) + 
					 torch.mean(pos_tail_emb ** 2)) / 3

		neg_regul = (torch.mean(neg_head_emb ** 2) + 
					 torch.mean(neg_relation_emb ** 2) + 
					 torch.mean(neg_tail_emb ** 2)) / 3

		regul = (pos_regul + neg_regul) / 2

		return regul

	def l3_regularization(self):

		"""L3 正则化函数，在损失函数中用到。

		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

def get_distmult_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`DistMult` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'DistMult'
			},
			'dim': {
				'values': [50, 100, 200]
			}
		}

	:returns: :py:class:`DistMult` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'DistMult'
		},
		'dim': {
			'values': [50, 100, 200]
		}
	}
		
	return parameters_dict