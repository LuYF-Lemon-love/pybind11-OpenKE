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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from ..utils.Timer import Timer

class TrainerDataParallel(object):

	"""
	:py:class:`TrainerDataParallel` 主要用于 KGE 模型的并行训练。
	"""

	def __init__(self,
		gpu_id,
		model = None,
		data_loader = None,
		train_times = 1000,
		alpha = 0.5,
		opt_method = "sgd",
		log_interval = None,
		save_interval = None,
		save_path = None):

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
			if self.log_interval and (epoch + 1) % self.log_interval == 0:
				print(f"[GPU{self.gpu_id}] Epoch [{epoch+1:>4d}/{self.train_times:>4d}] | Batchsize: {self.data_loader.batch_size} | Steps: {self.data_loader.nbatches} | loss: {res:>7f} | {timer.avg():.5f} sec/epoch")
			if self.gpu_id == 0 and self.save_interval and self.save_path and (epoch + 1) % self.save_interval == 0:
				path = os.path.join(os.path.splitext(self.save_path)[0] + "-" + str(epoch+1) + os.path.splitext(self.save_path)[-1])
				self.model.module.save_checkpoint(path)
				print(f"[GPU{self.gpu_id}] Epoch {epoch+1} | Training checkpoint saved at {path}")
		print(f"[GPU{self.gpu_id}] The model training is completed, taking a total of {timer.sum():.5f} seconds.")

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

def train(rank, world_size, model, data_loader, train_times, alpha, opt_method, log_interval, save_interval, save_path):

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
	:param log_interval: 训练几轮输出一次日志
	:type log_interval: int
	:param save_interval: 训练几轮保存一次模型
	:type save_interval: int
	:param save_path: 模型保存的路径
	:type save_path: str
	"""
	
	ddp_setup(rank, world_size)
	trainer = TrainerDataParallel(rank, model, data_loader, train_times, alpha, opt_method, log_interval, save_interval, save_path)
	trainer.run()
	destroy_process_group()
	
def trainer_distributed_data_parallel(model, data_loader, train_times, alpha, opt_method, log_interval, save_interval, save_path):

	"""生成进程。
	py:mod:`torch.multiprocessing` 是 Python 原生 ``multiprocessing`` 的一个 ``PyTorch`` 的包装器。
	``multiprocessing`` 的生成进程函数必须由 ``if __name__ == '__main__'`` 保护。

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
	:param log_interval: 训练几轮输出一次日志
	:type log_interval: int
	:param save_interval: 训练几轮保存一次模型
	:type save_interval: int
	:param save_path: 模型保存的路径
	:type save_path: str
	"""
	
	world_size = torch.cuda.device_count()
	mp.spawn(train, args = (world_size, model, data_loader, train_times, alpha, opt_method, log_interval, save_interval, save_path),
		nprocs = world_size)