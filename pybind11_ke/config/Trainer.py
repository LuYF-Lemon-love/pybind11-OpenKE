# coding:utf-8
#
# pybind11_ke/config/Trainer.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 29, 2023
#
# 该脚本定义了训练循环基类类.

"""
Trainer - 训练循环类。
"""

import os
import dgl
import wandb
import typing
import torch
from .Tester import Tester
import torch.optim as optim
from ..utils.Timer import Timer
from ..module.model import Model
from torch.utils.data import DataLoader
from ..utils.EarlyStopping import EarlyStopping
from ..module.strategy import NegativeSampling, RGCNSampling, CompGCNSampling

class Trainer(object):

	"""
	主要用于 KGE 模型的训练。

	例子::

		from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
		from pybind11_ke.module.model import TransE
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		from pybind11_ke.config import Trainer, Tester
		
		# dataloader for training
		dataloader = KGEDataLoader(
			in_path = "../../benchmarks/FB15K/", 
			batch_size = 8192,
			neg_ent = 25,
			test = True,
			test_batch_size = 256,
			num_workers = 16,
			train_sampler = BernSampler,
			test_sampler = TradTestSampler
		)
		
		# define the model
		transe = TransE(
			ent_tol = dataloader.train_sampler.ent_tol,
			rel_tol = dataloader.train_sampler.rel_tol,
			dim = 50, 
			p_norm = 1, 
			norm_flag = True)
		
		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			regul_rate = 0.01
		)
			
		# test the model
		tester = Tester(model = transe, data_loader = dataloader, use_gpu = True, device = 'cuda:1')
		
		# train the model
		trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
			epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100,
			save_path = '../../checkpoint/transe.pth', delta = 0.01)
		trainer.run()
	"""

	def __init__(
		self,
		model: NegativeSampling | RGCNSampling | CompGCNSampling | None = None,
		data_loader: typing.Union[DataLoader, None] = None,
		epochs: int = 1000,
		lr: float = 0.5,
		opt_method: str = "Adam",
		accelerator: str = None,
		use_gpu: bool = True,
		device: str = "cuda:0",
		tester: Tester | None = None,
		test: bool = False,
		valid_interval: int | None = None,
		log_interval: int | None = None,
		save_interval: int | None = None,
		save_path: str | None = None,
		use_early_stopping: bool = True,
		metric: str = 'hits@10',
		patience: int = 2,
		delta: float = 0,
		use_wandb: bool = False):

		"""创建 Trainer 对象。

		:param model: 包装 KGE 模型的训练策略类
		:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling` or :py:class:`pybind11_ke.module.strategy.RGCNSampling` or :py:class:`pybind11_ke.module.strategy.CompGCNSampling`
		:param data_loader: DataLoader
		:type data_loader: torch.utils.data.DataLoader
		:param epochs: 训练轮次数
		:type epochs: int
		:param lr: 学习率
		:type lr: float
		:param opt_method: 优化器: 'Adam' or 'adam', 'Adagrad' or 'adagrad', 'SGD' or 'sgd'
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
		:param use_early_stopping: 是否启用早停，需要 :py:attr:`tester` 和 :py:attr:`save_path` 不为空
		:type use_early_stopping: bool
		:param metric: 早停使用的验证指标，可选值：'mr', 'mrr', 'hits@N'。默认值：'hits@10'
		:type metric: str
		:param patience: :py:attr:`pybind11_ke.utils.EarlyStopping.patience` 参数，上次验证得分改善后等待多长时间。默认值：2
		:type patience: int
		:param delta: :py:attr:`pybind11_ke.utils.EarlyStopping.delta` 参数，监测数量的最小变化才符合改进条件。默认值：0
		:type delta: float
		:param use_wandb: 是否启用 wandb 进行日志输出
		:type use_wandb: bool
		"""
		
		#: 包装 KGE 模型的训练策略类，即 :py:class:`pybind11_ke.module.strategy.NegativeSampling` or :py:class:`pybind11_ke.module.strategy.RGCNSampling` or :py:class:`pybind11_ke.module.strategy.CompGCNSampling`
		self.model: NegativeSampling | RGCNSampling | CompGCNSampling = model

		#: :py:meth:`__init__` 传入的 :py:class:`torch.utils.data.DataLoader`
		self.data_loader: torch.utils.data.DataLoader = data_loader
		#: epochs
		self.epochs: int = epochs

		#: 学习率
		self.lr: float = lr
		#: 用户传入的优化器名字字符串
		self.opt_method: str = opt_method
		#: 根据 :py:meth:`__init__` 的 ``opt_method`` 生成对应的优化器
		self.optimizer: torch.optim.SGD | torch.optim.Adagrad | torch.optim.Adam | None = None
		#: 学习率调度器
		self.scheduler: torch.optim.lr_scheduler.MultiStepLR | None = None

		#: 是否进行分布式并行训练
		self.accelerator = accelerator
		
		#: 是否使用 gpu
		self.use_gpu: bool = use_gpu
		#: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
		self.device: typing.Union[torch.device, str] = torch.device(device) if self.use_gpu else "cpu"

		#: 用于模型评估的验证模型类
		self.tester: Tester | None = tester
		#: 是否在测试集上评估模型, :py:attr:`tester` 不为空
		self.test: bool = test
		#: 训练几轮在验证集上评估一次模型, :py:attr:`tester` 不为空
		self.valid_interval: int | None = valid_interval

		#: 训练几轮输出一次日志
		self.log_interval: int | None = log_interval
		#: 训练几轮保存一次模型
		self.save_interval: int | None = save_interval
		#: 模型保存的路径
		self.save_path: str | None = save_path

		#: 是否启用早停，需要 :py:attr:`tester` 和 :py:attr:`save_path` 不为空
		self.use_early_stopping: bool = use_early_stopping
		#: 早停使用的验证指标，可选值：'mr', 'mrr', 'hits@N'。默认值：'hits@10'
		self.metric: str = metric
		#: :py:attr:`pybind11_ke.utils.EarlyStopping.patience` 参数，上次验证得分改善后等待多长时间。默认值：2
		self.patience: int = patience
		#: :py:attr:`pybind11_ke.utils.EarlyStopping.delta` 参数，监测数量的最小变化才符合改进条件。默认值：0
		self.delta: float = delta
		#: 早停对象
		self.early_stopping: EarlyStopping = None

		#: 是否启用 wandb 进行日志输出
		self.use_wandb: bool = use_wandb

	def configure_optimizers(self):

		"""可以通过重新实现该方法自定义配置优化器。"""

		if self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.lr,
			)
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.lr,
			)
		elif self.opt_method == "SGD" or self.opt_method == "sgd":
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.lr,
				momentum=0.9,
			)

		if self.accelerator:
			self.optimizer = self.accelerator.prepare(self.optimizer)
			
		milestones = int(self.epochs / 3)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[milestones, milestones*2], gamma=0.1)

	def train_one_step(
		self,
		data: dict[str, typing.Union[str, dgl.DGLGraph, torch.Tensor]]) -> float:

		"""根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将
		模型训练 1 步。

		:param data: 训练数据
		:type data: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
		:returns: 损失值
		:rtype: float
		"""

		self.optimizer.zero_grad()
		if not self.accelerator:
			data = {key : self.to_var(value) if key != 'mode' else value for key, value in data.items()}
		loss = self.model(data)
		if not self.accelerator:
			loss.backward()
		else:
			self.accelerator.backward(loss)
		self.optimizer.step()		 
		return loss.item()

	def run(self):

		"""
		训练循环，首先根据 :py:attr:`use_gpu` 设置 :py:attr:`model` 是否使用 gpu 训练，然后根据
		:py:attr:`opt_method` 设置 :py:attr:`optimizer`，最后迭代 :py:attr:`data_loader` 获取数据，
		并利用 :py:meth:`train_one_step` 训练。
		"""

		if self.accelerator is None and self.use_gpu:
			self.model.cuda(device = self.device)

		if self.use_early_stopping and self.tester is not None and self.save_path is not None:
			self.early_stopping = EarlyStopping(
				save_path = os.path.split(self.save_path)[0],
				patience = self.patience,
				delta = self.delta)

		self.configure_optimizers()
		
		print(f"[{self.model.device}] Initialization completed, start model training.")
		
		if self.use_wandb:
			if not self.accelerator:
				wandb.watch(self.model.model, log_freq=100)
			else:
				wandb.watch(self.model.module.model, log_freq=100)
		
		timer = Timer()

		for epoch in range(self.epochs):

			res = 0.0
			if not self.accelerator:
				self.model.model.train()
			else:
				self.model.module.model.train()

			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			timer.stop()
			self.scheduler.step()

			if self.valid_interval and self.tester and (epoch + 1) % self.valid_interval == 0:
				print(f"[{self.model.device}] Epoch {epoch+1} | The model starts evaluation on the validation set.")
				self.print_test("link_valid", epoch)
			
			if self.early_stopping is not None and self.early_stopping.early_stop:
				print(f"[{self.model.device}] Early stopping")
				break
			
			if self.log_interval and (epoch + 1) % self.log_interval == 0:
				if self.use_wandb:
					wandb.log({"train/train_loss" : res, "train/epoch" : epoch + 1})
				print(f"[{self.model.device}] Epoch [{epoch+1:>4d}/{self.epochs:>4d}] | Batchsize: {self.data_loader.batch_size} | loss: {res:>9f} | {timer.avg():.5f} seconds/epoch")
			
			if (self.model.device.index == 0 or self.model.device.type == 'cpu') and self.save_interval and self.save_path and (epoch + 1) % self.save_interval == 0:
				path = os.path.join(os.path.splitext(self.save_path)[0] + "-" + str(epoch+1) + os.path.splitext(self.save_path)[-1])
				self.get_model().save_checkpoint(path)
				print(f"[{self.model.device}] Epoch {epoch+1} | Training checkpoint saved at {path}")
		
		print(f"[{self.model.device}] The model training is completed, taking a total of {timer.sum():.5f} seconds.")
		if self.use_wandb:
			wandb.log({"duration" : timer.sum()})
		
		if (self.model.device.index == 0 or self.model.device.type == 'cpu') and self.save_path:
			self.get_model().save_checkpoint(self.save_path)
			print(f"[{self.model.device}] Model saved at {self.save_path}.")
		
		if self.test and self.tester:
			print(f"[{self.model.device}] The model starts evaluating in the test set.")
			self.print_test("link_test")

	def print_test(
		self,
		sampling_mode: str,
		epoch: int = 0):

		"""根据 :py:attr:`tester` 类型进行链接预测 。

		:param sampling_mode: 数据
		:type sampling_mode: str
		"""

		self.tester.set_sampling_mode(sampling_mode)

		if sampling_mode == "link_test":
			mode = "test"
		elif sampling_mode == "link_valid":
			mode = "val"

		results = self.tester.run_link_prediction()
		for key, value in results.items():
			print(f"{key}: {value}")
		if self.use_wandb:
			log_dict = {f"{mode}/{key}" : value for key, value in results.items()}
			if sampling_mode == "link_valid":
				log_dict.update({
					"val/epoch": epoch
				})
			wandb.log(log_dict)
				
		if self.early_stopping is not None and sampling_mode == "link_valid":
			if self.metric in ['mr', 'mr_type']:
				self.early_stopping(-results[self.metric], self.get_model())
			elif self.metric in results.keys():
				self.early_stopping(results[self.metric], self.get_model())
			else:
				raise ValueError("Early stopping metric is not valid.")

	def to_var(
		self,
		x: torch.Tensor) -> torch.Tensor:

		"""将 ``x`` 转移到对应的设备上。

		:param x: 数据
		:type x: torch.Tensor
		:returns: 张量
		:rtype: torch.Tensor
		"""

		if self.use_gpu:
			return x.to(self.device)
		else:
			return x

	def get_model(self) -> Model:

		"""返回原始的 KGE 模型"""

		if self.accelerator:
			return self.model.module.model
		else:
			return self.model.model

def get_trainer_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`Trainer` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'trainer': {
				'value': 'Trainer'
			},
			'epochs': {
				'value': 10000
			},
			'lr': {
				'distribution': 'uniform',
				'min': 1e-5,
				'max': 1.0
			},
			'opt_method': {
				'values': ['adam', 'adagrad', 'sgd']
			},
			'valid_interval': {
				'value': 100
			},
			'log_interval': {
				'value': 100
			},
			'save_path': {
				'value': './model.pth'
			},
			'use_early_stopping': {
				'value': True
			},
			'metric': {
				'value': 'hits@10'
			},
			'patience': {
				'value': 2
			},
			'delta': {
				'value': 0.0001
			},
		}

	:returns: :py:class:`Trainer` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]	
	"""

	parameters_dict = {
		'trainer': {
			'value': 'Trainer'
		},
		'epochs': {
			'value': 10000
		},
		'lr': {
			'distribution': 'uniform',
			'min': 1e-5,
			'max': 1.0
		},
		'opt_method': {
			'values': ['adam', 'adagrad', 'sgd']
		},
		'valid_interval': {
			'value': 100
		},
		'log_interval': {
			'value': 100
		},
		'save_path': {
			'value': './model.pth'
		},
		'use_early_stopping': {
			'value': True
		},
		'metric': {
			'value': 'hits@10'
		},
		'patience': {
			'value': 2
		},
		'delta': {
			'value': 0.0001
		},
	}
		
	return parameters_dict