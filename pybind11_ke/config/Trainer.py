# coding:utf-8
#
# pybind11_ke/config/Trainer.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 5, 2023
#
# 该脚本定义了训练循环类.

"""
Trainer - 训练循环类，内部使用 ``tqmn`` 实现进度条。

基本用法如下：

.. code-block:: python

	# Import Trainer
	from openke.config import Trainer
	
	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 1.0, use_gpu = True)
	trainer.run()
"""

import torch
from torch.autograd import Variable
import torch.optim as optim
import os
from tqdm import tqdm

class Trainer(object):

	"""
	:py:class:`Trainer` 主要用于 KGE 模型的训练。
	"""

	def __init__(self,
		model = None,
		data_loader = None,
		train_times = 1000,
		alpha = 0.5,
		use_gpu = True,
		device = "cuda:0",
		opt_method = "sgd",
		save_interval = None,
		checkpoint_dir = None):

		"""创建 Trainer 对象。

		:param model: 包装 KGE 模型的训练策略类
		:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		:param data_loader: TrainDataLoader
		:type data_loader: :py:class:`pybind11_ke.data.TrainDataLoader`
		:param train_times: 训练轮次数
		:type train_times: int
		:param alpha: 学习率
		:type alpha: float
		:param use_gpu: 是否使用 gpu
		:type use_gpu: bool
		:param device: 使用哪个 gpu
		:type device: str
		:param opt_method: 优化器: Adagrad or adagrad, Adadelta or adadelta, Adam or adam, SGD or sgd
		:type opt_method: str
		:param save_interval: 训练几轮保存一次模型
		:type save_interval: int
		:param checkpoint_dir: 模型保存的目录
		:type checkpoint_dir: str
		"""

		#: epochs
		self.train_times = train_times

		#: 用户传入的优化器名字字符串
		self.opt_method = opt_method
		#: 根据 :py:meth:`__init__` 的 ``opt_method`` 生成对应的优化器
		self.optimizer = None
		#: 用于 :py:class:`torch.optim.Adagrad`
		self.lr_decay = 0
		#: 所有优化器（:py:class:`torch.optim.Adagrad`，:py:class:`torch.optim.Adadelta`，
		#: :py:class:`torch.optim.Adam`，:py:class:`torch.optim.SGD`）都可以设置
		self.weight_decay = 0
		#: 学习率
		self.alpha = alpha

		#: 包装 KGE 模型的训练策略类，即 :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		self.model = model

		#: :py:meth:`__init__` 传入的 :py:class:`pybind11_ke.data.TrainDataLoader`
		self.data_loader = data_loader

		#: 是否使用 gpu
		self.use_gpu = use_gpu
		#: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
		self.device = torch.device(device)

		#: 训练几轮保存一次模型
		self.save_interval = save_interval

		#: 模型保存的目录
		self.checkpoint_dir = checkpoint_dir

	def train_one_step(self, data):

		"""根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将
		模型训练 1 步。

		:param data: :py:attr:`data_loader` 利用
		 			 :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 函数生成的数据
		:type data: dict
		:returns: 损失值
		:rtype: float
		"""
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	def run(self):

		"""训练循环，首先根据 :py:attr:`use_gpu` 设置 :py:attr:`model` 是否处于 gpu，然后根据
		:py:attr:`opt_method` 设置 :py:attr:`optimizer`，最后迭代 :py:attr:`data_loader` 获取数据，
		并利用 :py:meth:`train_one_step` 训练。
		"""

		if self.use_gpu:
			self.model.cuda(device = self.device)

		if self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "SGD" or self.opt_method == "sgd":
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_interval and self.checkpoint_dir and (epoch + 1) % self.save_interval == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".pth"))

	def to_var(self, x, use_gpu):

		"""根据 ``use_gpu`` 返回 ``x`` 的张量

		:param x: 数据
		:type x: numpy.ndarray
		:param use_gpu: 是否使用 gpu
		:type use_gpu: bool
		:returns: 张量
		:rtype: torch.Tensor
		"""

		if use_gpu:
			return Variable(torch.from_numpy(x).cuda(device = self.device))
		else:
			return Variable(torch.from_numpy(x))