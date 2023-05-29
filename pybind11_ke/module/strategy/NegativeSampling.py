# coding:utf-8
#
# pybind11_ke/module/strategy/NegativeSampling.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
#
# 该脚本定义了 KGE 模型的训练策略.

"""
NegativeSampling - 训练策略类，包含损失函数。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer
	from pybind11_ke.module.loss import MarginLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 1.0, use_gpu = True)
"""

from .Strategy import Strategy

class NegativeSampling(Strategy):
	"""
	NegativeSampling 类，继承自 :py:class:`pybind11_ke.module.strategy.Strategy`。
	
	配和损失函数完成模型学习。
	"""

	def __init__(self, model = None, loss = None, batch_size = 256,
	      regul_rate = 0.0, l3_regul_rate = 0.0):
		"""创建 NegativeSampling 对象。

		:param model: KGE 模型
		:type model: :py:class:`pybind11_ke.module.model.Model`
		:param loss: 损失函数。
		:type loss: :py:class:`pybind11_ke.module.loss.Loss`
		:param batch_size: batch size
		:type batch_size: int
		:param regul_rate: 权重衰减系数
		:type regul_rate: float
		:param l3_regul_rate: l3 正则化系数
		:type l3_regul_rate: float
		"""

		super(NegativeSampling, self).__init__()
		#: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
		self.model = model
		#: 损失函数，即 :py:class:`pybind11_ke.module.loss.Loss`
		self.loss = loss
		#: batch size
		self.batch_size = batch_size
		#: 权重衰减系数
		self.regul_rate = regul_rate
		#: l3 正则化系数
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		"""
		获得正样本的得分，由于底层 C++ 处理模块的原因，所以正样本的
		的得分处于前 batch size 位置。

		:param score: 所有样本的得分。
		:type n_score: torch.Tensor
		:returns: 正样本的得分
		:rtype: torch.Tensor
		"""

		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		"""
		获得负样本的得分，由于底层 C++ 处理模块的原因，所以正样本的
		的得分处于前 batch size 位置，负样本处于正样本后面。

		:param score: 所有样本的得分。
		:type n_score: torch.Tensor
		:returns: 负样本的得分
		:rtype: torch.Tensor
		"""
				
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		"""计算最后的损失值。定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据
		:type data: dict
		:returns: 损失值
		:rtype: torch.Tensor
		"""

		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res