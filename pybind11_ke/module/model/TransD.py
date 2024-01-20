# coding:utf-8
#
# pybind11_ke/module/model/TransD.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 8, 2023
# 
# 该头文件定义了 TransD.

"""
TransD - 自动生成映射矩阵，简单而且高效，是对 TransR 的改进。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from typing_extensions import override

class TransD(Model):

	"""
	``TransD`` :cite:`TransD` 提出于 2015 年，自动生成映射矩阵，简单而且高效，是对 TransR 的改进。

	评分函数为:

	.. math::
	
		\parallel (\mathbf{r}_p \mathbf{h}_p^T + \mathbf{I})\mathbf{h} + \mathbf{r} - (\mathbf{r}_p \mathbf{t}_p^T + \mathbf{I})\mathbf{t} \parallel_{L_1/L_2}

	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`TransD <transd>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import TransD
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling

		# define the model
		transd = TransD(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim_e = config.dim_e, 
			dim_r = config.dim_r, 
			p_norm = config.p_norm, 
			norm_flag = config.norm_flag)

		# define the loss function
		model = NegativeSampling(
			model = transd, 
			loss = MarginLoss(margin = config.margin),
			batch_size = train_dataloader.get_batch_size()
		)

		# dataloader for test
		test_dataloader = TestDataLoader(in_path = config.in_path)

		# test the model
		tester = Tester(model = transd, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)

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
		dim_e: int = 100,
		dim_r: int = 100,
		p_norm: int = 1,
		norm_flag: bool = True):
		
		"""创建 TransD 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim_e: 实体嵌入和实体投影向量的维度
		:type dim_e: int
		:param dim_r: 关系嵌入和关系投影向量的维度
		:type dim_r: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		"""

		super(TransD, self).__init__(ent_tol, rel_tol)
		
		#: 实体嵌入和实体投影向量的维度
		self.dim_e: int = dim_e
		#: 关系嵌入和关系投影向量的维度
		self.dim_r: int = dim_r
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm: int = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag: bool = norm_flag

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim_e)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim_r)
		#: 根据实体个数，创建的实体投影向量
		self.ent_transfer: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim_e)
		#: 根据关系个数，创建的关系投影向量
		self.rel_transfer: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim_r)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_transfer.weight.data)
		nn.init.xavier_uniform_(self.rel_transfer.weight.data)

	@override
	def forward(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
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
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)
		score = self._calc(h ,t, r, mode)
		return score

	def _transfer(
		self,
		e: torch.Tensor,
		e_transfer: torch.Tensor,
		r_transfer: torch.Tensor) -> torch.Tensor:

		"""
		将头实体或尾实体的向量映射到关系向量空间。
		
		:param e: 头实体或尾实体向量。
		:type e: torch.Tensor
		:param e_transfer: 头实体或尾实体的投影向量
		:type e_transfer: torch.Tensor
		:param r_transfer: 关系的投影向量
		:type r_transfer: torch.Tensor
		:returns: 投影后的实体向量
		:rtype: torch.Tensor
		"""

		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], e.shape[-1])
			e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
			r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
			e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)			
			return e.view(-1, e.shape[-1])
		else:
			return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)

	def _resize(
		self,
		tensor: torch.Tensor,
		axis: int,
		size: int) -> torch.Tensor:

		"""计算实体向量与单位矩阵的乘法，并返回结果向量。

		源代码使用 :py:func:`torch.narrow` 进行缩小向量，
		:py:func:`torch.nn.functional.pad` 进行填充向量。
		
		:param tensor: 实体向量。
		:type tensor: torch.Tensor
		:param axis: 在哪个轴上进行乘法运算。
		:type axis: int
		:param size: 运算结果在 ``axis`` 上的维度大小，一般为关系向量的维度。
		:type size: int
		:returns: 乘法结果的向量
		:rtype: torch.Tensor
		"""

		shape = tensor.size()
		osize = shape[axis]
		if osize == size:
			return tensor
		if (osize > size):
			return torch.narrow(tensor, axis, 0, size)
		paddings = []
		for i in range(len(shape)):
			if i == axis:
				paddings = [0, size - osize] + paddings
			else:
				paddings = [0, 0] + paddings
		print (paddings)
		return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

	def _calc(
		self,
		h: torch.Tensor,
		t: torch.Tensor,
		r: torch.Tensor,
		mode: str) -> torch.Tensor:

		"""计算 TransD 的评分函数。
		
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
		
		# 保证 h, r, t 都是三维的
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		
		# 两者结果一样，括号只是逻辑上的，'head_batch' 是替换 head，否则替换 tail
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		
		# 利用距离函数计算得分
		score = torch.norm(score, self.p_norm, -1).flatten()
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
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) + 
				 torch.mean(h_transfer ** 2) + 
				 torch.mean(t_transfer ** 2) + 
				 torch.mean(r_transfer ** 2)) / 6
		return regul

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]]) -> np.ndarray:

		"""TransD 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = self.forward(data)
		return score.cpu().data.numpy()