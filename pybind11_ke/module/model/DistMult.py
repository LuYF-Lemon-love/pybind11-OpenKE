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
import numpy as np
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

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import DistMult
		from pybind11_ke.module.loss import SoftplusLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		distmult = DistMult(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = config.dim
		)
		
		# define the loss function
		model = NegativeSampling(
			model = distmult, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = config.regul_rate
		)
		
		# test the model
		tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)
		
		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader, epochs = config.epochs,
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

	def _calc(
		self,
		h: torch.Tensor,
		t: torch.Tensor,
		r: torch.Tensor,
		mode: str) -> torch.Tensor:

		"""计算 DistMult 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系的对角矩阵。
		:type r: torch.Tensor
		:param mode: ``normal`` 表示 :py:class:`pybind11_ke.data.TrainDataLoader` 
					 为训练同时进行头实体和尾实体负采样的数据，``head_batch`` 和 ``tail_batch`` 
					 表示为了减少数据传输成本，需要进行广播的数据，在广播前需要 reshape。
		:type mode: str
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		# 保证 h, r, t 都是三维的
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])

		# 两者结果一样，括号只是逻辑上的，'head_batch' 是替换 head，否则替换 tail
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t

		# 计算得分
		score = torch.sum(score, -1).flatten()
		return score

	@override
	def forward(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor, str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
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

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):

		"""L3 正则化函数，在损失函数中用到。

		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]]) -> np.ndarray:

		"""DistMult 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()
