# coding:utf-8
#
# pybind11_ke/module/model/Analogy.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 13, 2023
# 
# 该头文件定义了 Analogy.

"""
Analogy 类 - DistMult、HolE 和 ComplEx 的集大成者，效果与 HolE、ComplEx 差不多。
"""

import torch
import typing
import numpy as np
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class Analogy(Model):

	"""
	``Analogy`` :cite:`ANALOGY` 提出于 2017 年，:py:class:`pybind11_ke.module.model.DistMult`、:py:class:`pybind11_ke.module.model.HolE` 和 :py:class:`pybind11_ke.module.model.ComplEx` 的集大成者，
	效果与 :py:class:`pybind11_ke.module.model.HolE`、:py:class:`pybind11_ke.module.model.ComplEx` 差不多。 

	评分函数为:

	.. math::

	    <\operatorname{Re}(\mathbf{h_c}),\operatorname{Re}(\mathbf{r_c}),\operatorname{Re}(\mathbf{t_c})>
	        +<\operatorname{Re}(\mathbf{h_c}),\operatorname{Im}(\mathbf{r_c}),\operatorname{Im}(\mathbf{t_c})>
	        +<\operatorname{Im}(\mathbf{h_c}),\operatorname{Re}(\mathbf{r_c}),\operatorname{Im}(\mathbf{t_c})>
	        -<\operatorname{Im}(\mathbf{h_c}),\operatorname{Im}(\mathbf{r_c}),\operatorname{Re}(\mathbf{t_c})>
	        +<\mathbf{h_d}, \mathbf{r_d}, \mathbf{t_d}>

	评分函数为 :py:class:`pybind11_ke.module.model.DistMult` 和 :py:class:`pybind11_ke.module.model.ComplEx` 两者评分函数的和。:math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >` 为逐元素多线性点积（element-wise multi-linear dot product），
	正三元组的评分函数的值越大越好，负三元组越小越好，如果想获得更详细的信息请访问 :ref:`ANALOGY <analogy>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import Analogy
		from pybind11_ke.module.loss import SoftplusLoss
		from pybind11_ke.module.strategy import NegativeSampling

		# define the model
		analogy = Analogy(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = 200
		)

		# define the loss function
		model = NegativeSampling(
			model = analogy, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 1.0
		)

		# test the model
		tester = Tester(model = analogy, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			epochs = 2000, lr = 0.5, opt_method = "adagrad", use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100,
			save_path = '../../checkpoint/analogy.pth', delta = 0.01)
		trainer.run()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim: int = 100):

		"""创建 Analogy 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体嵌入向量和关系嵌入向量的维度
		:type dim: int
		"""

		super(Analogy, self).__init__(ent_tol, rel_tol)

		#: 实体嵌入向量和关系嵌入向量的维度
		self.dim: int = dim
		#: 根据实体个数，创建的实体嵌入的实部
		self.ent_re_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
		#: 根据实体个数，创建的实体嵌入的虚部
		self.ent_im_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
		#: 根据关系个数，创建的关系嵌入的实部
		self.rel_re_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)
		#: 根据关系个数，创建的关系嵌入的虚部
		self.rel_im_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)
		#: 根据实体个数，创建的实体嵌入，维度为 2 * :py:attr:`dim`
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim * 2)
		#: 根据关系个数，创建的关系嵌入, 维度为 2 * :py:attr:`dim`
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim * 2)
		
		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def _calc(
		self,
		h_re: torch.Tensor,
		h_im: torch.Tensor,
		h: torch.Tensor,
		t_re: torch.Tensor,
		t_im: torch.Tensor,
		t: torch.Tensor,
		r_re: torch.Tensor,
		r_im: torch.Tensor,
		r: torch.Tensor) -> torch.Tensor:

		"""计算 Analogy 的评分函数。
		
		:param h_re: 头实体的实部向量。
		:type h_re: torch.Tensor
		:param h_im: 头实体的虚部向量。
		:type h_im: torch.Tensor
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t_re: 尾实体的实部向量。
		:type t_re: torch.Tensor
		:param t_im: 尾实体的虚部向量。
		:type t_im: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r_re: 关系的实部向量。
		:type r_re: torch.Tensor
		:param r_im: 关系的虚部向量。
		:type r_im: torch.Tensor
		:param r: 关系的向量。
		:type r: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		return (torch.sum(r_re * h_re * t_re +
						   r_re * h_im * t_im +
						   r_im * h_re * t_im -
						   r_im * h_im * t_re, -1)
				+ torch.sum(h * t * r, -1))

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
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		h = self.ent_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		t = self.ent_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
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
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		h = self.ent_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		t = self.ent_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h_re ** 2) + 
				 torch.mean(h_im ** 2) + 
				 torch.mean(h ** 2) + 
				 torch.mean(t_re ** 2) + 
				 torch.mean(t_im ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r_re ** 2) + 
				 torch.mean(r_im ** 2) + 
				 torch.mean(r ** 2)) / 9
		return regul

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]]) -> np.ndarray:

		"""Analogy 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()

def get_analogy_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`Analogy` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'Analogy'
			},
			'dim': {
				'values': [50, 100, 200]
			}
		}

	:returns: :py:class:`Analogy` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'Analogy'
		},
		'dim': {
			'values': [50, 100, 200]
		}
	}
		
	return parameters_dict