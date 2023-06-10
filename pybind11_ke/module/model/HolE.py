# coding:utf-8
#
# pybind11_ke/module/model/HolE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 10, 2023
# 
# 该头文件定义了 HolE.

"""
HolE - 利用循环相关进行知识图谱嵌入，是 RESCAL 的压缩版本，因此非常容易的应用于大型的知识图谱。

论文地址: `Holographic Embeddings of Knowledge Graphs <https://ojs.aaai.org/index.php/AAAI/article/view/10314>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import HolE
	from pybind11_ke.module.loss import SoftplusLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the model
	hole = HolE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 100
	)

	# define the loss function
	model = NegativeSampling(
		model = hole, 
		loss = SoftplusLoss(),
		batch_size = train_dataloader.get_batch_size(), 
		regul_rate = 1.0
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	hole.save_checkpoint('../checkpoint/hole.ckpt')

	# test the model
	hole.load_checkpoint('../checkpoint/hole.ckpt')
	tester = Tester(model = hole, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class HolE(Model):

	"""
	HolE 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	HolE 提出于 2016 年，利用循环相关进行知识图谱嵌入，是 RESCAL 的压缩版本，因此非常容易的应用于大型的知识图谱。

	评分函数为: :math:`\mathbf{r}^T (\mathcal{F}^{-1}(\bar{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{b})))`，
	:math:`\mathcal{F}(\cdot)` 和 :math:`\mathcal{F}^{-1}(\cdot)` 表示快速傅里叶变换，
	:math:`\bar{\mathbf{x}}` 表示复数共轭，
	:math:`\odot` 表示哈达玛积。
	正三元组的评分函数的值越小越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):

		"""创建 HolE 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		:param margin: 原论文中损失函数的 gamma。
		:type margin: float
		:param epsilon: 对于 HolE 没什么用
		:type epsilon: float
		"""

		super(HolE, self).__init__(ent_tot, rel_tot)

		#: 实体和关系嵌入向量的维度
		self.dim = dim
		#: 原论文中损失函数的 gamma。
		self.margin = margin
		#: 对于 HolE 没什么用
		self.epsilon = epsilon

		#: 根据实体个数，创建的实体嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		#: 根据关系个数，创建的关系嵌入
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
	
	def _conj(self, tensor):
		"""获得复数共轭。

		tensor 减去 2 倍的虚部得到共轭复数。

		:param tensor: 复数。
		:type tensor: torch.Tensor
		:returns: 返回共轭复数。
		:rtype: torch.Tensor
		"""

		zero_shape = (list)(tensor.shape)
		one_shape = (list)(tensor.shape)
		zero_shape[-1] = 1
		one_shape[-1] -= 1
		ze = torch.zeros(size = zero_shape, device = tensor.device)
		on = torch.ones(size = one_shape, device = tensor.device)
		matrix = torch.cat([ze, on], -1)
		matrix = 2 * matrix
		return tensor - matrix * tensor
	
	def _real(self, tensor):
		"""获得复数的实部。

		利用 :py:func:`torch.narrow` 获得复数的实部。

		:param tensor: 复数。
		:type tensor: torch.Tensor
		:returns: 返回复数的实部。
		:rtype: torch.Tensor
		"""

		dimensions = len(tensor.shape)
		return tensor.narrow(dimensions - 1, 0, 1)

	def _imag(self, tensor):
		"""获得复数的虚部。

		利用 :py:func:`torch.narrow` 获得复数的虚部。

		:param tensor: 复数。
		:type tensor: torch.Tensor
		:returns: 返回复数的虚部。
		:rtype: torch.Tensor
		"""

		dimensions = len(tensor.shape)
		return tensor.narrow(dimensions - 1, 1, 1)

	def _mul(self, real_1, imag_1, real_2, imag_2):
		"""复数的乘法。

		公式为: :math:`(a+b\mathcal{i}) \cdot (c+d\mathcal{i}) = (ac-bd)+(ad+bc)\mathcal{i}`。

		:param a: 头实体的向量。
		:type a: torch.Tensor
		:param b: 尾实体的向量。
		:type b: torch.Tensor
		:returns: 返回循环相关计算结果。
		:rtype: torch.Tensor
		"""

		real = real_1 * real_2 - imag_1 * imag_2
		imag = real_1 * imag_2 + imag_1 * real_2
		return torch.cat([real, imag], -1)

	def _ccorr(self, a, b):
		"""计算循环相关 :math:`\mathcal{F}^{-1}(\bar{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{b}))`。
		
		利用 :py:func:`torch.rfft` 计算实数到复数离散傅里叶变换，
		利用 :py:func:`torch.ifft` 计算复数到复数离散傅立叶逆变换。

		:param a: 头实体的向量。
		:type a: torch.Tensor
		:param b: 尾实体的向量。
		:type b: torch.Tensor
		:returns: 返回循环相关计算结果。
		:rtype: torch.Tensor
		"""

		a = self._conj(torch.rfft(a, signal_ndim = 1, onesided = False))
		b = torch.rfft(b, signal_ndim = 1, onesided = False)
		res = self._mul(self._real(a), self._imag(a), self._real(b), self._imag(b))
		res = torch.ifft(res, signal_ndim = 1)
		return self._real(res).flatten(start_dim = -2)

	def _calc(self, h, t, r, mode):
		"""计算 HolE 的评分函数。
		
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

		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		score = self._ccorr(h, t) * r
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
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()
