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
		bern_flag = 1, 
		filter_flag = 1, 
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

	def __init__(self, in_path = "./", tri_file = None, ent_file = None,
		rel_file = None, batch_size = None, nbatches = None, threads = 8,
		sampling_mode = "normal", bern_flag = False, filter_flag = True,
		neg_ent = 1, neg_rel = 0):

		"""创建 TrainDataLoader 对象。

		:param in_path: 数据集目录
		:type in_path: str
		:param tri_file: train2id.txt
		:type tri_file: str
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
		:param sampling_mode: 数据采样模式，``normal`` 表示正常负采样，否则交替替换 head 和 tail 进行负采样
		:type sampling_mode: str
		:param bern_flag: 是否使用 TransH 提出的负采样方法进行负采样
		:type bern_flag: int
		:param filter_flag: 提出于 TransE，用于更好的构建负三元组，源代码一直使用，因此此开关不起作用
		:type filter_flag: bool
		:param neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		:type neg_ent: int
		:param neg_rel: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		:type neg_rel: int
		"""
		
		#: 数据集目录
		self.in_path = in_path
		#: train2id.txt
		self.tri_file = tri_file
		#: entity2id.txt
		self.ent_file = ent_file
		#: relation2id.txt
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"

		#: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供
		self.batch_size = batch_size
		#: nbatches 可以根据 batch_size 计算得出，两者不可以同时不提供
		self.nbatches = nbatches
		#: 底层 C++ 数据处理所需要的线程数
		self.work_threads = threads
		#: 数据采样模式，``normal`` 表示正常负采样，否则交替替换 head 和 tail 进行负采样
		self.sampling_mode = sampling_mode
		#: 是否使用 TransH 提出的负采样方法进行负采样
		self.bern = bern_flag
		#: 提出于 TransE，用于更好的构建负三元组，源代码一直使用，因此此开关不起作用
		self.filter = filter_flag
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		self.negative_ent = neg_ent
		#: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		self.negative_rel = neg_rel
		
		self.cross_sampling_flag = 0

		#: 实体的个数
		self.entTotal = 0
		#: 关系的个数
		self.relTotal = 0
		#: 训练集三元组的个数
		self.tripleTotal = 0

		# 读入数据
		self.read()

	def read(self):

		"""利用 ``pybind11`` 让底层 C++ 模块读取数据集中的数据"""
		
		if self.in_path != None:
			base.setInPath(self.in_path)
		else:
			base.setTrainPath(self.tri_file)
			base.setEntPath(self.ent_file)
			base.setRelPath(self.rel_file)
		
		base.setBern(self.bern)
		base.setWorkThreads(self.work_threads)
		base.randReset()
		base.importTrainFiles()

		# 实体的个数
		self.entTotal = base.getEntityTotal()
		# 关系的个数
		self.relTotal = base.getRelationTotal()
		# 训练集三元组的个数
		self.tripleTotal = base.getTrainTotal()

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

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
			self.batch_size, self.negative_ent, self.negative_rel, 0,
			self.filter, 0, 0)
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
			self.batch_size, self.negative_ent, self.negative_rel, -1,
			self.filter, 0, 0)
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
			self.batch_size, self.negative_ent, self.negative_rel, 1,
			self.filter, 0, 0)
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

	def set_in_path(self, in_path):

		"""设置 :py:attr:`in_path`
		
		:param in_path: 数据集目录
		:type in_path: str
		"""

		self.in_path = in_path

	def set_batch_size(self, batch_size):

		"""设置 :py:attr:`batch_size`
		
		:param batch_size: batch_size 可以根据 nbatches 计算得出，两者不可以同时不提供
		:type batch_size: int
		"""

		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_nbatches(self, nbatches):

		"""设置 :py:attr:`nbatches`
		
		:param nbatches: nbatches
		:type nbatches: int
		"""

		self.nbatches = nbatches
	
	def set_work_threads(self, work_threads):

		"""设置 :py:attr:`work_threads`
		
		:param work_threads: 底层 C++ 数据处理所需要的线程数
		:type work_threads: int
		"""

		self.work_threads = work_threads

	def set_bern_flag(self, bern):

		"""设置 :py:attr:`bern`
		
		:param bern: 是否使用 TransH 提出的负采样方法进行负采样
		:type bern: int
		"""

		self.bern = bern

	def set_filter_flag(self, filter):

		"""设置 :py:attr:`filter`
		
		:param filter: 提出于 TransE，用于更好的构建负三元组，源代码一直使用，因此此开关不起作用
		:type filter: bool
		"""

		self.filter = filter

	def set_ent_neg_rate(self, rate):

		"""设置 :py:attr:`negative_ent`
		
		:param rate: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail)
		:type rate: int
		"""

		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):

		"""设置 :py:attr:`negative_rel`
		
		:param rate: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation
		:type rate: int
		"""

		self.negative_rel = rate

	"""interfaces to get essential parameters"""

	def get_batch_size(self):

		"""返回 :py:attr:`batch_size`

		:returns: :py:attr:`batch_size`
		:rtype: int
		"""

		return self.batch_size

	def get_ent_tot(self):

		"""返回 :py:attr:`entTotal`

		:returns: :py:attr:`entTotal`
		:rtype: int
		"""

		return self.entTotal

	def get_rel_tot(self):

		"""返回 :py:attr:`relTotal`

		:returns: :py:attr:`relTotal`
		:rtype: int
		"""

		return self.relTotal

	def get_triple_tot(self):

		"""返回 :py:attr:`tripleTotal`

		:returns: :py:attr:`tripleTotal`
		:rtype: int
		"""

		return self.tripleTotal

	def __iter__(self):

		"""迭代器函数 :py:meth:`iterator.__iter__`，
		根据 :py:attr:`sampling_mode` 选择返回 :py:meth:`sampling` 和
		:py:meth:`cross_sampling`"""

		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		
		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`nbatches`
		:rtype: int
		"""

		return self.nbatches