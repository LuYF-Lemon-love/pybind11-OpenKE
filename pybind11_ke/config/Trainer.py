# coding:utf-8
#
# pybind11_ke/config/Trainer.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 5, 2023
#
# 该脚本定义了训练循环类.

"""
Trainer - 训练循环类。

基本用法如下：

.. code-block:: python

	# Import Trainer
	from pybind11_ke.config import Trainer
	
	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 0.01, use_gpu = True, device = 'cuda:1',
		log_interval = 100, save_interval = 100, save_path = "../../checkpoint/transe.pth")
	trainer.run()
"""

import torch
import torch.optim as optim
import os
from ..utils.Timer import Timer

class Trainer(object):

	"""
	:py:class:`Trainer` 主要用于 KGE 模型的训练。
	"""

	def __init__(self,
		model = None,
		data_loader = None,
		train_times = 1000,
		alpha = 0.5,
		opt_method = "sgd",
		use_gpu = True,
		device = "cuda:0",
		tester = None,
		test = False,
		valid_interval = None,
		log_interval = None,
		save_interval = None,
		save_path = None):

		"""创建 Trainer 对象。

		:param model: 包装 KGE 模型的训练策略类
		:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		:param data_loader: TrainDataLoader
		:type data_loader: :py:class:`pybind11_ke.data.TrainDataLoader`
		:param train_times: 训练轮次数
		:type train_times: int
		:param alpha: 学习率
		:type alpha: float
		:param opt_method: 优化器: Adam or adam, SGD or sgd
		:type opt_method: str
		:param use_gpu: 是否使用 gpu
		:type use_gpu: bool
		:param device: 使用哪个 gpu
		:type device: str
		:param tester: 用于模型评估的验证模型类
		:type tester: :py:class:`pybind11_ke.config.Tester`
		:param test: 是否在测试集上评估模型, :py:attr:`tester` 不为空
		:type test: bool
		:param valid_interval: 训练几轮在验证集上评估一次模型, :py:attr:`tester` 不为空
		:type valid_interval: int
		:param log_interval: 训练几轮输出一次日志
		:type log_interval: int
		:param save_interval: 训练几轮保存一次模型
		:type save_interval: int
		:param save_path: 模型保存的路径
		:type save_path: str
		"""

		#: 包装 KGE 模型的训练策略类，即 :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		self.model = model

		#: :py:meth:`__init__` 传入的 :py:class:`pybind11_ke.data.TrainDataLoader`
		self.data_loader = data_loader
		#: epochs
		self.train_times = train_times

		#: 学习率
		self.alpha = alpha
		#: 用户传入的优化器名字字符串
		self.opt_method = opt_method
		#: 根据 :py:meth:`__init__` 的 ``opt_method`` 生成对应的优化器
		self.optimizer = None

		#: 是否使用 gpu
		self.use_gpu = use_gpu
		#: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
		self.device = torch.device(device)

		#: 用于模型评估的验证模型类
		self.tester = tester
		#: 是否在测试集上评估模型, :py:attr:`tester` 不为空
		self.test = test
		#: 训练几轮在验证集上评估一次模型, :py:attr:`tester` 不为空
		self.valid_interval = valid_interval

		#: 训练几轮输出一次日志
		self.log_interval = log_interval
		#: 训练几轮保存一次模型
		self.save_interval = save_interval
		#: 模型保存的路径
		self.save_path = save_path

	def train_one_step(self, data):

		"""根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将
		模型训练 1 步。

		:param data: :py:attr:`data_loader` 利用 :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 函数生成的数据
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

		if self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
			)
		elif self.opt_method == "SGD" or self.opt_method == "sgd":
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
			)
		print("Finish initializing...")
		
		timer = Timer()
		for epoch in range(self.train_times):
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			timer.stop()
			if self.valid_interval and self.tester and (epoch + 1) % self.valid_interval == 0:
				print(f"[{self.device}] Epoch {epoch+1} | The model starts evaluation on the validation set.")
				self.tester.set_sampling_mode("link_valid")
				self.tester.run_link_prediction()
			if self.log_interval and (epoch + 1) % self.log_interval == 0:
				print(f"[{self.device}] Epoch [{epoch+1:>4d}/{self.train_times:>4d}] | Batchsize: {self.data_loader.batch_size} | Steps: {self.data_loader.nbatches} | loss: {res:>9f} | {timer.avg():.5f} seconds/epoch")
			if self.save_interval and self.save_path and (epoch + 1) % self.save_interval == 0:
				path = os.path.join(os.path.splitext(self.save_path)[0] + "-" + str(epoch+1) + os.path.splitext(self.save_path)[-1])
				self.model.model.save_checkpoint(path)
				print(f"[{self.device}] Epoch {epoch+1} | Training checkpoint saved at {path}")
		print(f"[{self.device}] The model training is completed, taking a total of {timer.sum():.5f} seconds.")
		if self.save_path:
			self.model.model.save_checkpoint(self.save_path)
			print(f"[{self.device}] Model saved at {self.save_path}.")
		if self.test and self.tester:
			print(f"[{self.device}] The model starts evaluating in the test set.")
			self.tester.set_sampling_mode("link_test")
			self.tester.run_link_prediction()

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
			return torch.from_numpy(x).to(self.device)
		else:
			return torch.from_numpy(x)