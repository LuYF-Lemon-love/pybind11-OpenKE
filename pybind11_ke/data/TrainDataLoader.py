# coding:utf-8
#
# pybind11_ke/data/TrainDataLoader.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 3, 2023
#
# 该脚本定义了采样数据的函数.

"""
TrainDataLoader - 数据集类，类似 :py:class:`torch.utils.data.DataLoader`。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer
	from pybind11_ke.module.model import TransE
	from pybind11_ke.module.loss import MarginLoss
	from pybind11_ke.module.strategy import NegativeSampling
	from pybind11_ke.data import TrainDataLoader

	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "../benchmarks/FB15K237/", 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = True, 
		neg_ent = 25,
		neg_rel = 0)

	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 200, 
		p_norm = 1, 
		norm_flag = True)

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

import numpy as np
import base

class TrainDataSampler(object):

	"""将 :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
	或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling` 
	包装起来。
	"""

	def __init__(self, nbatches, datasampler):

		"""创建 TrainDataSample 对象。
		
		:param nbatches: 1 epoch 有多少个 batch
		:type nbatches: int
		:param datasampler: 采样器
		:type datasampler: :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
							或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling`
		"""

		#: 1 epoch 有多少个 batch
		self.nbatches = nbatches
		#: :py:meth:`pybind11_ke.data.TrainDataLoader.sampling` 
		#: 或 :py:meth:`pybind11_ke.data.TrainDataLoader.cross_sampling` 函数
		self.datasampler = datasampler
		self.batch = 0

	def __iter__(self):

		"""迭代器函数 :py:meth:`iterator.__iter__`"""

		return self

	def __next__(self):

		"""迭代器函数 :py:meth:`iterator.__next__`"""

		self.batch += 1 
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.datasampler()

	def __len__(self):

		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`nbatches`
		:rtype: int
		"""

		return self.nbatches

class TrainDataLoader(object):

	"""
	:py:class:`TrainDataLoader` 主要从底层 C++ 模块获得数据用于 KGE 模型的训练。
	"""

	def __init__(self, in_path = "./", train_file = "train2id.txt", ent_file = "entity2id.txt",
		rel_file = "relation2id.txt", batch_size = None, nbatches = None, threads = 8,
		sampling_mode = "normal", bern_flag = False,
		neg_ent = 1, neg_rel = 0):

		"""创建 TrainDataLoader 对象。

		:param in_path: 数据集目录
		:type in_path: str
		:param train_file: train2id.txt
		:type train_file: str
		:param ent_file: entity2id.txt
		:type ent_file: str
		:param rel_file: relation2id.txt
		:type rel_file: str
		:param batch_size: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供
		:type batch_size: int
		:param nbatches: nbatches 可以根据 batch_size 计算得出，两者不可以同时不提供
		:type nbatches: int
		:param threads: 底层 C++ 数据处理所需要的线程数
		:type threads: int
		:param sampling_mode: 数据采样模式，``normal`` 表示正常负采样，``cross`` 表示交替替换 head 和 tail 进行负采样
		:type sampling_mode: str
		:param bern_flag: 是否使用 TransH 提出的负采样方法进行负采样
		:type bern_flag: bool
		:param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		:type neg_ent: int
		:param neg_rel: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		:type neg_rel: int
		"""
		
		#: 数据集目录
		self.in_path = in_path
		#: train2id.txt
		self.train_file = self.in_path + train_file
		#: entity2id.txt
		self.ent_file = self.in_path + ent_file
		#: relation2id.txt
		self.rel_file = self.in_path + rel_file

		#: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供
		self.batch_size = batch_size
		#: nbatches 可以根据 batch_size 计算得出，两者不可以同时不提供
		self.nbatches = nbatches
		#: 底层 C++ 数据处理所需要的线程数
		self.threads = threads
		#: 数据采样模式，``normal`` 表示正常负采样，``cross`` 表示交替替换 head 和 tail 进行负采样
		self.sampling_mode = sampling_mode
		#: 是否使用 TransH 提出的负采样方法进行负采样
		self.bern_flag = bern_flag
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		self.neg_ent = neg_ent
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		self.neg_rel = neg_rel
		
		self.cross_sampling_flag = 0

		#: 实体的个数
		self.ent_tol = 0
		#: 关系的个数
		self.rel_tol = 0
		#: 训练集三元组的个数
		self.train_tot = 0

		# 读入数据
		self.read()

	def read(self):

		"""利用 ``pybind11`` 让底层 C++ 模块读取数据集中的数据"""
		
		base.setInPath(self.in_path)
		base.setTrainPath(self.train_file)
		base.setEntPath(self.ent_file)
		base.setRelPath(self.rel_file)
		
		base.setBern(self.bern_flag)
		base.setWorkThreads(self.threads)
		base.randReset()
		base.importTrainFiles()

		# 实体的个数
		self.ent_tol = base.getEntityTotal()
		# 关系的个数
		self.rel_tol = base.getRelationTotal()
		# 训练集三元组的个数
		self.train_tot = base.getTrainTotal()

		if self.batch_size == None:
			self.batch_size = self.train_tot // self.nbatches
		if self.nbatches == None:
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
			self.batch_size, self.neg_ent, self.neg_rel, 0,
			0)
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
			self.batch_size, self.neg_ent, self.neg_rel, -1,
			0)
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
			self.batch_size, self.neg_ent, self.neg_rel, 1,
			0)
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

	"""interfaces to get essential parameters"""

	def get_batch_size(self):

		"""返回 :py:attr:`batch_size`

		:returns: :py:attr:`batch_size`
		:rtype: int
		"""

		return self.batch_size

	def get_ent_tot(self):

		"""返回 :py:attr:`ent_tol`

		:returns: :py:attr:`ent_tol`
		:rtype: int
		"""

		return self.ent_tol

	def get_rel_tot(self):

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