# coding:utf-8
#
# pybind11_ke/config/TrainerDataParallel.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 5, 2023
#
# 该脚本定义了数据并行训练循环类.

"""
TrainerDataParallel - 数据并行训练循环类，内部使用 ``tqmn`` 实现进度条。

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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

class TrainerDDP(object):

	"""
	:py:class:`TrainerDDP` 主要用于 KGE 模型的训练。
	"""

	def __init__(self,
		gpu_id,
		model = None,
		data_loader = None,
		train_times = 1000,
		alpha = 0.5,
		opt_method = "sgd",
		save_steps = None,
		checkpoint_dir = None):

		"""创建 Trainer 对象。

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
		:param opt_method: 优化器: Adagrad or adagrad, Adadelta or adadelta, Adam or adam, SGD or sgd
		:type opt_method: str
		:param save_steps: 训练几轮保存一次模型
		:type save_steps: int
		:param checkpoint_dir: 模型保存的目录
		:type checkpoint_dir: str
		"""

		#: 第几个 gpu
		self.gpu_id = gpu_id
		
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
		self.model = model.to(gpu_id)

		#: :py:meth:`__init__` 传入的 :py:class:`pybind11_ke.data.TrainDataLoader`
		self.data_loader = data_loader

		#: 训练几轮保存一次模型
		self.save_steps = save_steps

		#: 模型保存的目录
		self.checkpoint_dir = checkpoint_dir
		
		self.model = DDP(self.model, device_ids=[gpu_id])

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
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			if self.save_steps and (epoch + 1) % self.save_steps == 0:
				print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.data_loader.batch_size} | Steps: {self.data_loader.nbatches}")
				if self.gpu_id == 0 and self.checkpoint_dir:
					path = os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".pth")
					self.model.module.save_checkpoint(path)
					print(f"[GPU{self.gpu_id}] Epoch {epoch} | Training checkpoint saved at {path}")

	def to_var(self, x):

		"""根据 ``use_gpu`` 返回 ``x`` 的张量

		:param x: 数据
		:type x: numpy.ndarray
		:returns: 张量
		:rtype: torch.Tensor
		"""

		return torch.from_numpy(x).to(self.gpu_id)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
def train(rank, world_size, model, data_loader, train_times, alpha, opt_method, save_steps, checkpoint_dir):
	ddp_setup(rank, world_size)
	trainer = TrainerDDP(rank, model, data_loader, train_times, alpha, opt_method, save_steps, checkpoint_dir)
	trainer.run()
	destroy_process_group()
def TrainerDataParallel(model, data_loader, train_times, alpha, opt_method, save_steps, checkpoint_dir):
	world_size = torch.cuda.device_count()
	mp.spawn(train, args = (world_size, model, data_loader, train_times, alpha, opt_method, save_steps, checkpoint_dir), nprocs = world_size)