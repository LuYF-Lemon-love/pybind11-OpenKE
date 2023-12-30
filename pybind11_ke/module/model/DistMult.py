# coding:utf-8
#
# pybind11_ke/module/model/DistMult.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 5, 2023
# 
# 该头文件定义了 DistMult.

"""
:py:class:`DistMult` 类 - 最简单的双线性模型，与 TransE 参数量相同，因此非常容易的应用于大型的知识图谱。

论文地址: `Embedding Entities and Relations for Learning and Inference in Knowledge Bases <https://arxiv.org/abs/1412.6575>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import DistMult
	from pybind11_ke.module.loss import SoftplusLoss
	from pybind11_ke.module.strategy import NegativeSampling
	
	# define the model
	distmult = DistMult(
		ent_tot = train_dataloader.get_ent_tol(),
		rel_tot = train_dataloader.get_rel_tol(),
		dim = 200
	)
	
	# define the loss function
	model = NegativeSampling(
		model = distmult, 
		loss = SoftplusLoss(),
		batch_size = train_dataloader.get_batch_size(), 
		regul_rate = 1.0
	)
	
	
	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 2000, lr = 0.5, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	distmult.save_checkpoint('../checkpoint/distmult.ckpt')
	
	# test the model
	distmult.load_checkpoint('../checkpoint/distmult.ckpt')
	tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class DistMult(Model):

	"""
	:py:class:`DistMult` 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	DistMult 提出于 2015 年，最简单的双线性模型，与 TransE 参数量相同，
	因此非常容易的应用于大型的知识图谱。

	评分函数为: :math:`< \mathbf{h}, \mathbf{r}, \mathbf{t} >`，
	为逐元素多线性点积（element-wise multi-linear dot product），
	正三元组的评分函数的值越大越好，负三元组越小越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):

		"""创建 DistMult 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体嵌入向量和关系对角矩阵的维度
		:type dim: int
		:param margin: 原论文中损失函数的 gamma，值为 1，但是该参数对于模型训练没用。
		:type margin: float
		:param epsilon: 对于 DistMult 没什么用
		:type epsilon: float
		"""

		super(DistMult, self).__init__(ent_tot, rel_tot)

		#: 实体嵌入向量和关系对角矩阵的维度
		self.dim = dim
		#: 原论文中损失函数的 gamma，值为 1，但是该参数对于模型训练没用。
		self.margin = margin
		#: 对于 DistMult 没什么用
		self.epsilon = epsilon
		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		#: 根据关系个数，创建的关系对角矩阵
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h, t, r, mode):
		"""计算 DistMult 的评分函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系的对角矩阵。
		:type r: torch.Tensor
		:param mode: 如果进行链接预测的话：``normal`` 表示 :py:class:`pybind11_ke.data.TrainDataLoader` 
					 为训练进行采样的数据，``head_batch`` 和 ``tail_batch`` 
					 表示 :py:class:`pybind11_ke.data.TestDataLoader` 为验证模型采样的数据。
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
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):
		"""L3 正则化函数，在损失函数中用到。

		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		"""DistMult 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()
