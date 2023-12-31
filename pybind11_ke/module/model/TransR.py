# coding:utf-8
#
# pybind11_ke/module/model/TransR.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 31, 2023
# 
# 该头文件定义了 TransR.

"""
TransR - 是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

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
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
			dim = 100, 
			p_norm = 1, 
			norm_flag = True)

		transr = TransR(
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
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

	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1,
		norm_flag = True, rand_init = False):

		"""创建 TransR 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
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
		"""

		super(TransR, self).__init__(ent_tot, rel_tot)
		
		#: 实体嵌入向量的维度
		self.dim_e = dim_e
		#: 关系嵌入向量的维度
		self.dim_r = dim_r
		#: 评分函数的距离函数, 按照原论文，这里可以取 1 或 2。
		self.p_norm = p_norm
		#: 是否利用 :py:func:`torch.nn.functional.normalize` 
		#: 对实体和关系嵌入向量的最后一维执行 L2-norm。
		self.norm_flag = norm_flag
		#: 关系矩阵是否采用随机初始化
		self.rand_init = rand_init

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		#: 根据关系个数，创建的关系嵌入
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

		self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)

		if not self.rand_init:
			identity = torch.zeros(self.dim_e, self.dim_r)
			for i in range(min(self.dim_e, self.dim_r)):
				identity[i][i] = 1
			identity = identity.view(self.dim_e * self.dim_r)
			for i in range(self.rel_tot):
				self.transfer_matrix.weight.data[i] = identity
		else:
			nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

	def _calc(self, h, t, r, mode):

		"""计算 TransR 的评分函数。
		
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
	
	def _transfer(self, e, r_transfer):

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
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
			e = torch.matmul(e, r_transfer).permute(1, 0, 2)
		else:
			e = e.view(-1, 1, self.dim_e)
			e = torch.matmul(e, r_transfer)
		return e.view(-1, self.dim_r)

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
		r_transfer = self.transfer_matrix(batch_r)
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)
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
		r_transfer = self.transfer_matrix(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_transfer ** 2)) / 4
		return regul

	def predict(self, data):

		"""TransR 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = self.forward(data)
		return score.cpu().data.numpy()