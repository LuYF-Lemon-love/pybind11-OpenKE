# coding:utf-8
#
# pybind11_ke/data/TrainDataLoader.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 27, 2023
#
# 该脚本定义了为训练循环采样数据的类.

"""
TrainDataLoader - 数据集类，类似 :py:class:`torch.utils.data.DataLoader`。
"""

import base
import numpy as np

class TrainDataSampler(object):

	"""将 :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
	或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling` 
	进行封装。
	"""

	def __init__(self, nbatches, sampler):

		"""创建 TrainDataSample 对象。
		
		:param nbatches: 1 epoch 有多少个 batch
		:type nbatches: int
		:param sampler: 采样器
		:type sampler: :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
							或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling`
		"""

		#: 1 epoch 有多少个 batch
		self.nbatches = nbatches
		#: :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
		#: 或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling` 函数
		self.sampler = sampler
		self.batch = 0

	def __iter__(self):

		"""迭代器函数 :py:meth:`iterator.__iter__`"""

		return self

	def __next__(self):

		"""迭代器函数 :py:meth:`iterator.__next__`"""

		self.batch += 1 
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.sampler()

	def __len__(self):

		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`nbatches`
		:rtype: int
		"""

		return self.nbatches

class TrainDataLoader(object):

	"""
	主要从底层 C++ 模块获得数据用于 KGE 模型的训练。
	
	例子::

		from pybind11_ke.config import Trainer
		from pybind11_ke.module.model import TransE
		from pybind11_ke.module.loss import MarginLoss
		from pybind11_ke.module.strategy import NegativeSampling
		from pybind11_ke.data import TrainDataLoader

		# dataloader for training
		train_dataloader = TrainDataLoader(
			in_path = "../../benchmarks/FB15K/", 
			nbatches = 200,
			threads = 8, 
			sampling_mode = "normal", 
			bern = False,  
			neg_ent = 25,
			neg_rel = 0)

		# define the model
		transe = TransE(
			ent_tot = train_dataloader.get_ent_tol(),
			rel_tot = train_dataloader.get_rel_tol(),
			dim = 50, 
			p_norm = 1, 
			norm_flag = True)

		# define the loss function
		model = NegativeSampling(
			model = transe, 
			loss = MarginLoss(margin = 1.0),
			batch_size = train_dataloader.get_batch_size()
		)

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
			train_times = 1000, alpha = 0.01, use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transe.pth')
		trainer.run()
	"""

	def __init__(self, in_path = "./", ent_file = "entity2id.txt", rel_file = "relation2id.txt",
		train_file = "train2id.txt", batch_size = None, nbatches = None, threads = 8,
		sampling_mode = "normal", bern = True,
		neg_ent = 1, neg_rel = 0):

		"""创建 TrainDataLoader 对象。

		:param in_path: 数据集目录
		:type in_path: str
		:param ent_file: entity2id.txt
		:type ent_file: str
		:param rel_file: relation2id.txt
		:type rel_file: str
		:param train_file: train2id.txt
		:type train_file: str
		:param batch_size: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供；同时指定时 batch_size 优先级更高
		:type batch_size: int
		:param nbatches: nbatches 可以根据 batch_size 计算得出，两者不可以同时不提供；同时指定时 batch_size 优先级更高
		:type nbatches: int
		:param threads: 底层 C++ 数据处理所需要的线程数
		:type threads: int
		:param sampling_mode: 数据采样模式，``normal`` 表示正常负采样，``cross`` 表示交替替换 head 和 tail 进行负采样
		:type sampling_mode: str
		:param bern: 是否使用 TransH 提出的负采样方法进行负采样
		:type bern: bool
		:param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		:type neg_ent: int
		:param neg_rel: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		:type neg_rel: int
		"""
		
		#: 数据集目录
		self.in_path = in_path
		#: entity2id.txt
		self.ent_file = self.in_path + ent_file
		#: relation2id.txt
		self.rel_file = self.in_path + rel_file
		#: train2id.txt
		self.train_file = self.in_path + train_file

		#: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供；同时指定时 batch_size 优先级更高
		self.batch_size = batch_size
		#: nbatches 可以根据 batch_size 计算得出，两者不可以同时不提供；同时指定时 batch_size 优先级更高
		self.nbatches = nbatches
		#: 底层 C++ 数据处理所需要的线程数
		self.threads = threads
		#: 数据采样模式，``normal`` 表示正常负采样，``cross`` 表示交替替换 head 和 tail 进行负采样
		self.sampling_mode = sampling_mode
		#: 是否使用 TransH 提出的负采样方法进行负采样
		self.bern = bern
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		self.neg_ent = neg_ent
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		self.neg_rel = neg_rel

		#: 实体的个数
		self.ent_tol = 0
		#: 关系的个数
		self.rel_tol = 0
		#: 训练集三元组的个数
		self.train_tot = 0

		self.cross_sampling_flag = 0

		# 读入数据
		self.read()

	def read(self):

		"""利用 `pybind11 <https://github.com/pybind/pybind11>`__ 让底层 C++ 模块读取数据集中的数据"""

		print("\nStart reading training data...")
		
		base.set_in_path(self.in_path)
		base.set_ent_path(self.ent_file)
		base.set_rel_path(self.rel_file)
		base.set_train_path(self.train_file)
		
		base.set_bern(self.bern)
		base.set_work_threads(self.threads)
		base.rand_reset()
		base.read_train_files()

		print("Training data read completed.\n")

		# 实体的个数
		self.ent_tol = base.get_entity_total()
		# 关系的个数
		self.rel_tol = base.get_relation_total()
		# 训练集三元组的个数
		self.train_tot = base.get_train_total()

		if self.batch_size == None and self.nbatches == None:
			raise ValueError("batch_size or nbatches must specify one")
		elif self.batch_size == None:
			self.batch_size = self.train_tot // self.nbatches
		else:
			self.nbatches = self.train_tot // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.neg_ent + self.neg_rel)

		# 利用 np.zeros 分配内存
		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)

	def sampling(self):

		"""正常采样1 batch 数据，即 ``normal``
		
		:returns: 1 batch 数据
		:rtype: dict
		"""

		base.sampling(self.batch_h, self.batch_t, self.batch_r, self.batch_y,
			self.batch_size, self.neg_ent, self.neg_rel, 0)
		return {
			"batch_h": self.batch_h, 
			"batch_t": self.batch_t, 
			"batch_r": self.batch_r, 
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	def sampling_head(self):

		"""只替换 head 进行负采样, 生成 1 batch 数据

		:returns: 1 batch 数据
		:rtype: dict
		"""

		base.sampling(self.batch_h, self.batch_t, self.batch_r, self.batch_y,
			self.batch_size, self.neg_ent, self.neg_rel, 1)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	def sampling_tail(self):

		"""只替换 tail 进行负采样, 生成 1 batch 数据
		
		:returns: 1 batch 数据
		:rtype: dict
		"""

		base.sampling(self.batch_h, self.batch_t, self.batch_r, self.batch_y,
			self.batch_size, self.neg_ent, self.neg_rel, -1)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	def cross_sampling(self):

		"""交替替换 head 和 tail 进行负采样, 生成 1 batch 数据
		
		:returns: 1 batch 数据
		:rtype: dict
		"""

		self.cross_sampling_flag = 1 - self.cross_sampling_flag 
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	def get_batch_size(self):

		"""返回 :py:attr:`batch_size`

		:returns: :py:attr:`batch_size`
		:rtype: int
		"""

		return self.batch_size

	def get_ent_tol(self):

		"""返回 :py:attr:`ent_tol`

		:returns: :py:attr:`ent_tol`
		:rtype: int
		"""

		return self.ent_tol

	def get_rel_tol(self):

		"""返回 :py:attr:`rel_tol`

		:returns: :py:attr:`rel_tol`
		:rtype: int
		"""

		return self.rel_tol

	def get_train_tot(self):

		"""返回 :py:attr:`train_tot`

		:returns: :py:attr:`train_tot`
		:rtype: int
		"""

		return self.train_tot

	def __iter__(self):

		"""迭代器函数 :py:meth:`iterator.__iter__`，
		根据 :py:attr:`sampling_mode` 选择返回 :py:meth:`sampling` 和
		:py:meth:`cross_sampling`"""

		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		elif self.sampling_mode == "cross":
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		
		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`nbatches`
		:rtype: int
		"""

		return self.nbatches