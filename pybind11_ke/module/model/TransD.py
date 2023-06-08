# coding:utf-8
#
# pybind11_ke/module/model/TransD.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 8, 2023
# 
# 该头文件定义了 TransD.

"""
TransD - 自动生成映射矩阵，简单而且高效，是对 TransR 的改进。

论文地址: `Knowledge Graph Embedding via Dynamic Mapping Matrix <https://aclanthology.org/P15-1067/>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import TransD
	from pybind11_ke.module.loss import MarginLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the model
	transd = TransD(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim_e = 200, 
		dim_r = 200, 
		p_norm = 1, 
		norm_flag = True)


	# define the loss function
	model = NegativeSampling(
		model = transd, 
		loss = MarginLoss(margin = 4.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
	                  train_times = 1000, alpha = 1.0, use_gpu = True)
	trainer.run()
	transd.save_checkpoint('../checkpoint/transd.ckpt')

	# test the model
	transd.load_checkpoint('../checkpoint/transd.ckpt')
	tester = Tester(model = transd, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransD(Model):

	"""
	TransD 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	TransD 提出于 2015 年，自动生成映射矩阵，简单而且高效，是对 TransR 的改进。

	评分函数为: :math:`\parallel (\mathbf{r}_p \mathbf{h}_p^T + \mathbf(I))\mathbf{h} + \mathbf{r} - (\mathbf{r}_p \mathbf{t}_p^T + \mathbf(I))\mathbf{t} \parallel_{L_1/L_2}`，
	正三元组的评分函数的值越小越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1,
	      norm_flag = True, margin = None, epsilon = None):
		
		"""创建 TransD 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim_e: 实体嵌入和实体投影向量的维度
		:type dim_e: int
		:param dim_r: 关系嵌入和关系投影向量的维度
		:type dim_r: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		:param margin: 原论文中损失函数的 gamma。
		:type margin: float
		:param epsilon: 对于 TransD 没什么用
		:type epsilon: float
		"""

		super(TransD, self).__init__(ent_tot, rel_tot)
		
		#: 实体嵌入和实体投影向量的维度
		self.dim_e = dim_e
		#: 关系嵌入和关系投影向量的维度
		self.dim_r = dim_r
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag = norm_flag
		#: 原论文中损失函数的 gamma。
		self.margin = margin
		#: 对于 TransE 没什么用
		self.epsilon = epsilon

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
		#: 根据实体个数，创建的实体投影向量
		self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
		#: 根据关系个数，创建的关系投影向量
		self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.ent_transfer.weight.data)
			nn.init.xavier_uniform_(self.rel_transfer.weight.data)
		else:
			self.ent_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
			)
			self.rel_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.ent_embedding_range.item(), 
				b = self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent_transfer.weight.data, 
				a= -self.ent_embedding_range.item(), 
				b= self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_transfer.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			#: :py:attr:`margin` 是否为 None。
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _resize(self, tensor, axis, size):
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

	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def _transfer(self, e, e_transfer, r_transfer):
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

	def forward(self, data):
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
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
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

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()