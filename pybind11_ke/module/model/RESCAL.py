# coding:utf-8
#
# pybind11_ke/module/model/RESCAL.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 1, 2023
# 
# 该头文件定义了 RESCAL.

"""
:py:class:`RESCAL` 类 - 一个张量分解模型。

论文地址: `A Three-Way Model for Collective Learning on Multi-Relational Data <https://icml.cc/Conferences/2011/papers/438_icmlpaper.pdf>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import RESCAL
	from pybind11_ke.module.loss import MarginLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the model
	rescal = RESCAL(
		ent_tot = train_dataloader.get_ent_tol(),
		rel_tot = train_dataloader.get_rel_tol(),
		dim = 50
	)

	# define the loss function
	model = NegativeSampling(
		model = rescal, 
		loss = MarginLoss(margin = 1.0),
		batch_size = train_dataloader.get_batch_size(), 
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 0.1, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	rescal.save_checkpoint('../checkpoint/rescal.ckpt')

	# test the model
	rescal.load_checkpoint('../checkpoint/rescal.ckpt')
	tester = Tester(model = rescal, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class RESCAL(Model):

	"""
	:py:class:`RESCAL` 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	RESCAL 提出于 2011 年，是很多张量分解模型的基石，模型较复杂。

	评分函数为: :math:`-\mathbf{h}^T \mathbf{M}_r \mathbf{t}`，
	正三元组的评分函数的值越小越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100):

		"""创建 RESCAL 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		"""

		super(RESCAL, self).__init__(ent_tot, rel_tot)

		#: 实体和关系嵌入向量的维度
		self.dim = dim
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		#: 根据关系个数，创建的关系矩阵
		self.rel_matrices = nn.Embedding(self.rel_tot, self.dim * self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)
	
	def _calc(self, h, t, r):
		"""计算 RESCAL 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系矩阵。
		:type r: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		t = t.view(-1, self.dim, 1)
		r = r.view(-1, self.dim, self.dim)
		tr = torch.matmul(r, t)
		tr = tr.view(-1, self.dim)
		return -torch.sum(h * tr, -1)

	def forward(self, data):
		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		score = self._calc(h ,t, r)
		return score

	def regularization(self, data):
		"""L2 正则化函数（又称权重衰减），在损失函数中用到。
		
		:param data: 数据。
		:type data: dict
		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		"""RESCAL 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()