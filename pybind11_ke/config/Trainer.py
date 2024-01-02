# coding:utf-8
#
# pybind11_ke/config/Trainer.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
#
# 该脚本定义了训练循环类.

"""
Trainer - 训练循环类。
"""

import os
import wandb
import torch
import torch.optim as optim
from ..utils.Timer import Timer
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer(object):

	"""
	主要用于 KGE 模型的训练。

	例子::

		from pybind11_ke.config import Trainer

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transe.pth')
		trainer.run()
	"""

	def __init__(self,
		model = None,
		data_loader = None,
		epochs = 1000,
		lr = 0.5,
		opt_method = "Adam",
		use_gpu = True,
		device = "cuda:0",
		tester = None,
		test = False,
		valid_interval = None,
		log_interval = None,
		save_interval = None,
		save_path = None,
		use_wandb = False,
		gpu_id = None):

		"""创建 Trainer 对象。

		:param model: 包装 KGE 模型的训练策略类
		:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		:param data_loader: TrainDataLoader
		:type data_loader: :py:class:`pybind11_ke.data.TrainDataLoader`
		:param epochs: 训练轮次数
		:type epochs: int
		:param lr: 学习率
		:type lr: float
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
		:param use_wandb: 是否启用 wandb 进行日志输出
		:type use_wandb: bool
		:param gpu_id: 第几个 gpu，用于并行训练
		:type gpu_id: int
		"""

		#: 第几个 gpu
		self.gpu_id = gpu_id
		
		#: 包装 KGE 模型的训练策略类，即 :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		self.model = DDP(model.to(self.gpu_id), device_ids=[self.gpu_id]) if self.gpu_id is not None else model

		#: :py:meth:`__init__` 传入的 :py:class:`pybind11_ke.data.TrainDataLoader`
		self.data_loader = data_loader
		#: epochs
		self.epochs = epochs

		#: 学习率
		self.lr = lr
		#: 用户传入的优化器名字字符串
		self.opt_method = opt_method
		#: 根据 :py:meth:`__init__` 的 ``opt_method`` 生成对应的优化器
		self.optimizer = None
		#: 学习率调度器
		self.scheduler = None

		#: 是否使用 gpu
		self.use_gpu = use_gpu
		#: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
		self.device = torch.device(device) if self.use_gpu else "cpu"

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
		#: 是否启用 wandb 进行日志输出
		self.use_wandb = use_wandb

		self.print_device = f"GPU{self.gpu_id}" if self.gpu_id is not None else self.device

	def configure_optimizers(self):

		"""可以通过重新实现该方法自定义配置优化器。"""

		if self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.lr,
			)
		elif self.opt_method == "SGD" or self.opt_method == "sgd":
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.lr,
				momentum=0.9,
			)
			
		milestones = int(self.epochs / 3)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[milestones, milestones*2], gamma=0.1)

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
			'batch_h': self.to_var(data['batch_h']),
			'batch_t': self.to_var(data['batch_t']),
			'batch_r': self.to_var(data['batch_r']),
			'batch_y': self.to_var(data['batch_y']),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	def run(self):

		"""训练循环，首先根据 :py:attr:`use_gpu` 设置 :py:attr:`model` 是否使用 gpu 训练，然后根据
		:py:attr:`opt_method` 设置 :py:attr:`optimizer`，最后迭代 :py:attr:`data_loader` 获取数据，
		并利用 :py:meth:`train_one_step` 训练。
		"""

		if self.gpu_id is None and self.use_gpu:
			self.model.cuda(device = self.device)

		self.configure_optimizers()
		
		if self.gpu_id is not None or self.use_gpu:
			print(f"[{self.print_device}] Initialization completed, start model training.")
		else:
			print("Initialization completed, start model training.")
		
		timer = Timer()
		for epoch in range(self.epochs):
			res = 0.0
			if self.gpu_id is not None:
				self.model.module.model.train()
			else:
				self.model.model.train()
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			timer.stop()
			self.scheduler.step()
			if (self.gpu_id is None or self.gpu_id == 0) and self.valid_interval and self.tester and (epoch + 1) % self.valid_interval == 0:
				print(f"[{self.print_device}] Epoch {epoch+1} | The model starts evaluation on the validation set.")
				self.tester.set_sampling_mode("link_valid")
				if self.tester.data_loader.type_constrain:
					mr, mrr, hit1, hit3, hit10, mrTC, mrrTC, hit1TC, hit3TC, hit10TC = self.tester.run_link_prediction()
					wandb.log({
						"val/epoch": epoch,
						"val/mr" : mr,
						"val/mrr" : mrr,
						"val/hit1" : hit1,
						"val/hit3" : hit3,
						"val/hit10" : hit10,
						"val/mrTC" : mrTC,
						"val/mrrTC" : mrrTC,
						"val/hit1TC" : hit1TC,
						"val/hit3TC" : hit3TC,
						"val/hit10TC" : hit10TC,
					})
				else:
					mr, mrr, hit1, hit3, hit10 = self.tester.run_link_prediction()
					wandb.log({
						"val/epoch": epoch,
						"val/mr" : mr,
						"val/mrr" : mrr,
						"val/hit1" : hit1,
						"val/hit3" : hit3,
						"val/hit10" : hit10,
					})
			if self.log_interval and (epoch + 1) % self.log_interval == 0:
				if (self.gpu_id is None or self.gpu_id == 0) and self.use_wandb:
					wandb.log({"train/train_loss" : res, "train/epoch" : epoch + 1})
				print(f"[{self.print_device}] Epoch [{epoch+1:>4d}/{self.epochs:>4d}] | Batchsize: {self.data_loader.batch_size} | Steps: {self.data_loader.nbatches} | loss: {res:>9f} | {timer.avg():.5f} seconds/epoch")
			if (self.gpu_id is None or self.gpu_id == 0) and self.save_interval and self.save_path and (epoch + 1) % self.save_interval == 0:
				path = os.path.join(os.path.splitext(self.save_path)[0] + "-" + str(epoch+1) + os.path.splitext(self.save_path)[-1])
				if self.gpu_id == 0:
					self.model.module.model.save_checkpoint(path)
				else:
					self.model.model.save_checkpoint(path)
				print(f"[{self.print_device}] Epoch {epoch+1} | Training checkpoint saved at {path}")
		print(f"[{self.print_device}] The model training is completed, taking a total of {timer.sum():.5f} seconds.")
		if (self.gpu_id is None or self.gpu_id == 0) and self.save_path:
			if self.gpu_id == 0:
				self.model.module.model.save_checkpoint(self.save_path)
			else:
				self.model.model.save_checkpoint(self.save_path)
			print(f"[{self.print_device}] Model saved at {self.save_path}.")
		if (self.gpu_id is None or self.gpu_id == 0) and self.test and self.tester:
			print(f"[{self.print_device}] The model starts evaluating in the test set.")
			self.tester.set_sampling_mode("link_test")
			if self.tester.data_loader.type_constrain:
				mr, mrr, hit1, hit3, hit10, mrTC, mrrTC, hit1TC, hit3TC, hit10TC = self.tester.run_link_prediction()
				wandb.log({
					"test/mr" : mr,
					"test/mrr" : mrr,
					"test/hit1" : hit1,
					"test/hit3" : hit3,
					"test/hit10" : hit10,
					"test/mrTC" : mrTC,
					"test/mrrTC" : mrrTC,
					"test/hit1TC" : hit1TC,
					"test/hit3TC" : hit3TC,
					"test/hit10TC" : hit10TC,
				})
			else:
				mr, mrr, hit1, hit3, hit10 = self.tester.run_link_prediction()
				wandb.log({
					"test/mr" : mr,
					"test/mrr" : mrr,
					"test/hit1" : hit1,
					"test/hit3" : hit3,
					"test/hit10" : hit10,
				})
			self.tester.run_link_prediction()

	def to_var(self, x):

		"""将 ``x`` 转移到对应的设备上。

		:param x: 数据
		:type x: numpy.ndarray
		:returns: 张量
		:rtype: torch.Tensor
		"""

		if self.gpu_id is not None:
			return torch.from_numpy(x).to(self.gpu_id)
		elif self.use_gpu:
			return torch.from_numpy(x).to(self.device)
		else:
			return torch.from_numpy(x)