# coding:utf-8
#
# pybind11_ke/module/model/TransH.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
# 
# 该头文件定义了 TransH.

"""
TransH - 是第二个平移模型，将关系建模为超平面上的平移操作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransH(Model):

	"""
	``TransH`` :cite:`TransH` 提出于 2014 年，是第二个平移模型，将关系建模为超平面上的平移操作。
	
	评分函数为:
	
	.. math::

		\Vert (h-r_w^\top hr_w)+r_d-(t-r_w^\top tr_w)\Vert_{L_1/L_2}

	正三元组的评分函数的值越小越好，如果想获得更详细的信息请访问 :ref:`TransH <transh>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import TransH
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		
		# define the model
		transh = TransH(
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
			dim = 200, 
			p_norm = 1, 
			norm_flag = True)
		
		# define the loss function
		model = NegativeSampling(
			model = transh, 
			loss = MarginLoss(margin = 4.0),
			batch_size = train_dataloader.get_batch_size()
		)
		
		# test the model
		tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')
		
		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			epochs = 1000, lr = 0.5, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transe.pth')
		trainer.run()
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1,
	      norm_flag = True):
		
		"""创建 TransH 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体、关系嵌入向量和和法向量的维度
		:type dim: int
		:param p_norm: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		:type p_norm: int
		:param norm_flag: 是否利用 :py:func:`torch.nn.functional.normalize` 
						  对实体和关系嵌入的最后一维执行 L2-norm。
		:type norm_flag: bool
		"""

		super(TransH, self).__init__(ent_tot, rel_tot)
		
		#: 实体、关系嵌入向量和和法向量的维度
		self.dim = dim
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag = norm_flag
		
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		#: 根据关系个数，创建的法向量
		self.norm_vector = nn.Embedding(self.rel_tot, self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.norm_vector.weight.data)

	def _calc(self, h, t, r, mode):

		"""计算 TransH 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系实体的向量。
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

	def _transfer(self, e, norm):

		"""
		将头实体或尾实体的向量投影到超平面上。
		
		:param e: 头实体或尾实体向量。
		:type e: torch.Tensor
		:param norm: 法向量
		:type norm: torch.Tensor
		:returns: 投影后的实体向量
		:rtype: torch.Tensor
		"""

		norm = F.normalize(norm, p = 2, dim = -1)
		if e.shape[0] != norm.shape[0]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			e = e - torch.sum(e * norm, -1, True) * norm
			return e.view(-1, e.shape[-1])
		else:
			return e - torch.sum(e * norm, -1, True) * norm

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
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		h = self._transfer(h, r_norm)
		t = self._transfer(t, r_norm)
		score = self._calc(h ,t, r, mode)
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
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_norm ** 2)) / 4
		return regul
	
	def predict(self, data):

		"""TransE 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""
		
		score = self.forward(data)
		return score.cpu().data.numpy()