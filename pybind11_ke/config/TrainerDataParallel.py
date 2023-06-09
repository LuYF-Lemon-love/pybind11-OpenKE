# coding:utf-8
#
# pybind11_ke/config/TrainerDataParallel.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 6, 2023
#
# 该脚本定义了并行训练循环函数.

"""
:py:func:`trainer_distributed_data_parallel` - 并行训练循环函数。

基本用法如下：

.. code-block:: python

	# Import trainer_distributed_data_parallel
	from pybind11_ke.config import trainer_distributed_data_parallel
	
	# train the model
	if __name__ == "__main__":
		trainer_distributed_data_parallel(model = model, data_loader = tra
			train_times = 1000, alpha = 0.01, opt_method = "sgd", log_inte
			save_interval = 50, save_path = "../../checkpoint/transe.pth")
"""

import torch
import torch.optim as optim
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from ..utils.Timer import Timer
from .Tester import Tester
from ..data import TestDataLoader

class TrainerDataParallel(object):

	"""
	:py:class:`TrainerDataParallel` 主要用于 KGE 模型的并行训练。
	"""

	def __init__(self,
		gpu_id,
		model,
		data_loader,
		train_times,
		alpha,
		opt_method,
		tester,
		test,
		valid_interval,
		log_interval,
		save_interval,
		save_path):

		"""创建 TrainerDataParallel 对象。

		:param gpu_id: 第几个 gpu
		:type gpu_id: int
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

		#: 第几个 gpu
		self.gpu_id = gpu_id
		#: 包装 KGE 模型的训练策略类，即 :py:class:`pybind11_ke.module.strategy.NegativeSampling`
		self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
		
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
		
		print(f"[GPU{self.gpu_id}] Initialization completed, start model training.")

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

		"""训练循环，利用 :py:meth:`train_one_step` 训练。
		"""
		
		timer = Timer()
		for epoch in range(self.train_times):
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			timer.stop()
			if self.gpu_id == 0 and self.valid_interval and self.tester and (epoch + 1) % self.valid_interval == 0:
				print(f"[GPU{self.gpu_id}] Epoch {epoch+1} | The model starts evaluation on the validation set.")
				self.tester.set_sampling_mode("link_valid")
				self.tester.run_link_prediction()
			if self.log_interval and (epoch + 1) % self.log_interval == 0:
				print(f"[GPU{self.gpu_id}] Epoch [{epoch+1:>4d}/{self.train_times:>4d}] | Batchsize: {self.data_loader.batch_size} | Steps: {self.data_loader.nbatches} | loss: {res:>9f} | {timer.avg():.5f} seconds/epoch")
			if self.gpu_id == 0 and self.save_interval and self.save_path and (epoch + 1) % self.save_interval == 0:
				path = os.path.join(os.path.splitext(self.save_path)[0] + "-" + str(epoch+1) + os.path.splitext(self.save_path)[-1])
				self.model.module.model.save_checkpoint(path)
				print(f"[GPU{self.gpu_id}] Epoch {epoch+1} | Training checkpoint saved at {path}")
		print(f"[GPU{self.gpu_id}] The model training is completed, taking a total of {timer.sum():.5f} seconds.")
		if self.gpu_id == 0 and self.save_path:
			self.model.module.model.save_checkpoint(self.save_path)
			print(f"[GPU{self.gpu_id}] Model saved at {self.save_path}.")
		if  self.gpu_id == 0 and self.test and self.tester:
			print(f"[GPU{self.gpu_id}] The model starts evaluating in the test set.")
			self.tester.set_sampling_mode("link_test")
			self.tester.run_link_prediction()

	def to_var(self, x):

		"""返回 GPU 中 ``x`` 的副本。

		:param x: 数据
		:type x: numpy.ndarray
		:returns: 张量
		:rtype: torch.Tensor
		"""

		return torch.from_numpy(x).to(self.gpu_id)

def ddp_setup(rank, world_size):

	"""
	构建进程组。在 Windows 上， :py:mod:`torch.distributed` 仅支持 `Gloo` 后端。

	:param rank: 进程的唯一标识符
	:type rank: int
	:param world_size: 进程的总数
	:type world_size: int
	"""
	
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	init_process_group(backend="gloo", rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)

def train(rank,
	world_size,
	model,
	data_loader,
	train_times,
	alpha,
	opt_method,
	test,
	valid_interval,
	log_interval,
	save_interval,
	save_path,
	valid_file,
	test_file,
	type_constrain):

	"""进程函数。

	:param gpu_id: 第几个 gpu
	:type gpu_id: int
	:param world_size: 进程的总数
	:type world_size: int
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
	:param valid_file: valid2id.txt
	:type valid_file: str
	:param test_file: test2id.txt
	:type test_file: str
	:param type_constrain: 是否用 type_constrain.txt 进行负采样
	:type type_constrain: bool
	"""
	
	ddp_setup(rank, world_size)
	tester = None
	if test:
		test_dataloader = TestDataLoader(in_path = data_loader.in_path, ent_file = data_loader.ent_file,
			rel_file = data_loader.rel_file, train_file = data_loader.train_file, valid_file = valid_file,
			test_file = test_file, type_constrain = type_constrain, sampling_mode = 'link')
		tester = Tester(model = model.model, data_loader = test_dataloader)
	trainer = TrainerDataParallel(rank, model, data_loader, train_times, alpha, opt_method,
		tester, test, valid_interval, log_interval, save_interval, save_path)
	trainer.run()
	destroy_process_group()
	
def trainer_distributed_data_parallel(model = None,
	data_loader = None,
	train_times = 1000,
	alpha = 0.5,
	opt_method = "sgd",
	test = False,
	valid_interval = None,
	log_interval = None,
	save_interval = None,
	save_path = None,
	valid_file = "valid2id.txt",
	test_file = "test2id.txt",
	type_constrain = True):

	"""生成进程。
	py:mod:`torch.multiprocessing` 是 Python 原生 ``multiprocessing`` 的一个 ``PyTorch`` 的包装器。
	``multiprocessing`` 的生成进程函数必须由 ``if __name__ == '__main__'`` 保护。
	有效的 batch size 是 :py:attr:`pybind11_ke.data.TrainDataLoader.batch_size` * ``nprocs``。

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
	:param valid_file: valid2id.txt
	:type valid_file: str
	:param test_file: test2id.txt
	:type test_file: str
	:param type_constrain: 是否用 type_constrain.txt 进行负采样
	:type type_constrain: bool
	"""
	
	world_size = torch.cuda.device_count()
	mp.spawn(train, args = (world_size, model, data_loader, train_times, alpha, opt_method,
							test, valid_interval, log_interval, save_interval, save_path,
							valid_file, test_file, type_constrain),
				nprocs = world_size)