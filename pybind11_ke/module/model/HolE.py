# coding:utf-8
#
# pybind11_ke/module/model/HolE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 9, 2023
# 
# 该头文件定义了 HolE.

"""
HolE - 利用循环相关进行知识图谱嵌入，是 RESCAL 的压缩版本，因此非常容易的应用于大型的知识图谱。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class HolE(Model):

	"""
	``HolE`` :cite:`HolE` 提出于 2016 年，利用循环相关进行知识图谱嵌入，是 RESCAL 的压缩版本，因此非常容易的应用于大型的知识图谱。

	评分函数为:

	.. math::
	
		\mathbf{r}^T (\mathcal{F}^{-1}(\overline{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{t})))
	
	其中 :math:`\mathcal{F}(\cdot)` 和 :math:`\mathcal{F}^{-1}(\cdot)` 表示快速傅里叶变换，:math:`\overline{\mathbf{x}}` 表示复数共轭，:math:`\odot` 表示哈达玛积。
	
	正三元组的评分函数的值越大越好，负三元组越小越好，如果想获得更详细的信息请访问 :ref:`HolE <hole>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import HolE
		from pybind11_ke.module.loss import SoftplusLoss
		from pybind11_ke.module.strategy import NegativeSampling

		# define the model
		hole = HolE(
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
			dim = config.dim
		)

		# define the loss function
		model = NegativeSampling(
			model = hole, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = config.regul_rate
		)

		# test the model
		tester = Tester(model = hole, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)

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
		ent_tot: int,
		rel_tot: int,
		dim: int = 100):

		"""创建 HolE 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		"""

		super(HolE, self).__init__(ent_tot, rel_tot)

		#: 实体和关系嵌入向量的维度
		self.dim = dim

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def _ccorr(
		self,
		a: torch.Tensor,
		b: torch.Tensor) -> torch.Tensor:

		"""计算循环相关 :math:`\mathcal{F}^{-1}(\overline{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{t}))`。
		
		利用 :py:func:`torch.fft.rfft` 计算实数到复数离散傅里叶变换，:py:func:`torch.fft.irfft` 是其逆变换；
		利用 :py:func:`torch.conj` 计算复数的共轭。

		:param a: 头实体的向量。
		:type a: torch.Tensor
		:param b: 尾实体的向量。
		:type b: torch.Tensor
		:returns: 返回循环相关计算结果。
		:rtype: torch.Tensor
		"""
		
		# 计算傅里叶变换
		a_fft = torch.fft.rfft(a, dim=-1)
		b_fft = torch.fft.rfft(b, dim=-1)
		
		# 复数的共轭
		a_fft = torch.conj(a_fft)
		
		# 哈达玛积
		p_fft = a_fft * b_fft
    	
		# 傅里叶变换的逆变换
		return torch.fft.irfft(p_fft, n=a.shape[-1], dim=-1)

	def _calc(
		self,
		h: torch.Tensor,
		t: torch.Tensor,
		r: torch.Tensor,
		mode: str) -> torch.Tensor:

		"""计算 HolE 的评分函数。
		
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

		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		score = self._ccorr(h, t) * r
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

	def l3_regularization(self) -> torch.Tensor:

		"""L3 正则化函数，在损失函数中用到。

		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""
		
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)
	
	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> np.ndarray:
		
		"""HolE 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor, str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()
