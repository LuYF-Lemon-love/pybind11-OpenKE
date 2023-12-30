# coding:utf-8
#
# pybind11_ke/module/model/RotatE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 26, 2023
# 
# 该头文件定义了 RotatE.

"""
:py:class:`RotatE` 类 - 将实体表示成复数向量，关系建模为复数向量空间的旋转。

论文地址: `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space <https://openreview.net/forum?id=HkgEQnRqYQ>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import RotatE
	from pybind11_ke.module.loss import SigmoidLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the model
	rotate = RotatE(
		ent_tot = train_dataloader.get_ent_tol(),
		rel_tot = train_dataloader.get_rel_tol(),
		dim = 1024,
		margin = 6.0,
		epsilon = 2.0,
	)

	# define the loss function
	model = NegativeSampling(
		model = rotate, 
		loss = SigmoidLoss(adv_temperature = 2),
		batch_size = train_dataloader.get_batch_size(), 
		regul_rate = 0.0
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
	                  train_times = 6000, lr = 2e-5, use_gpu = True, opt_method = "adam")
	trainer.run()
	rotate.save_checkpoint('../checkpoint/rotate.ckpt')

	# test the model
	rotate.load_checkpoint('../checkpoint/rotate.ckpt')
	tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class RotatE(Model):

	"""
	:py:class:`RotatE` 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	RotatE 提出于 2019 年，将实体表示成复数向量，关系建模为复数向量空间的旋转。

	评分函数为: :math:`\gamma - \parallel \mathbf{h} \circ \mathbf{r} - \mathbf{t} \parallel_{L_2}`，
	:math:`\circ` 表示哈达玛积（Hadamard product），正三元组的评分函数的值越大越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100, margin = 6.0, epsilon = 2.0):

		"""创建 RotatE 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		:param margin: 原论文中损失函数的 gamma。
		:type margin: float
		:param epsilon: RotatE 原论文对应的源代码固定为 2.0。
		:type epsilon: float
		"""

		super(RotatE, self).__init__(ent_tot, rel_tot)

		#: RotatE 原论文对应的源代码固定为 2.0。
		self.epsilon = epsilon

		#: RotatE 原论文的实现中将实体嵌入向量的维度指定为 ``dim`` 的 2 倍。
		#: 因为实体嵌入向量需要划分为实部和虚部。
		self.dim_e = dim * 2
		#: 关系嵌入向量的维度，为 ``dim``。
		self.dim_r = dim

		#: 根据实体个数，创建的实体嵌入。
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		#: 根据关系个数，创建的关系嵌入。
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		#: 原论文中损失函数的 gamma。
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

	def _calc(self, h, t, r, mode):

		"""计算 RotatE 的评分函数。

		利用 :py:func:`torch.chunk` 拆分实体嵌入向量获得复数的实部和虚部。
		依据原论文使用的 L2-norm 作为距离函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:param r: 关系的向量。
		:type r: torch.Tensor
		:param mode: 如果进行链接预测的话：``normal`` 表示 :py:class:`pybind11_ke.data.TrainDataLoader` 
					 为训练进行采样的数据，``head_batch`` 和 ``tail_batch`` 
					 表示 :py:class:`pybind11_ke.data.TestDataLoader` 为验证模型采样的数据。
		:type mode: str
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		pi = self.pi_const

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		# Make phases of relations uniformly distributed in [-pi, pi]
		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

		if mode == "head_batch":
			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head
		else:
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score.permute(1, 0).flatten()

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
		score = self.margin - self._calc(h ,t, r, mode)
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
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):

		"""RotatE 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		score = -self.forward(data)
		return score.cpu().data.numpy()