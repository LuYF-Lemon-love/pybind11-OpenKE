# coding:utf-8
#
# pybind11-ke/data/TrainDataLoader.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 24, 2023
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
		in_path = "./benchmarks/FB15K237/", 
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

	"""将 :py:meth:`TrainDataLoader.sampling` 或 :py:meth:`TrainDataLoader.cross_sampling` 
	包装起来。
	"""

	def __init__(self, nbatches, datasampler):
		"""创建 TrainDataSample 对象
		
		:param nbatches: 1 epoch 有多少个 batch 
		:type nbatches: int
		:param datasampler: 采样器
		:type datasampler: :py:meth:`TrainDataLoader.sampling` 或 :py:meth:`TrainDataLoader.cross_sampling`
		"""

		#: 1 epoch 有多少个 batch
		self.nbatches = nbatches
		#: :py:meth:`TrainDataLoader.sampling` 或 :py:meth:`TrainDataLoader.cross_sampling` 函数
		self.datasampler = datasampler
		self.batch = 0

	def __iter__(self):
		"""迭代器函数 py:meth:`__iter__`"""

		return self

	def __next__(self):
		"""迭代器函数 __next__"""

		self.batch += 1 
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.datasampler()

	def __len__(self):
		"""len() 要求"""

		return self.nbatches

class TrainDataLoader(object):

	def __init__(self, 
		in_path = "./",
		tri_file = None,
		ent_file = None,
		rel_file = None,
		batch_size = None,
		nbatches = None,
		threads = 8,
		sampling_mode = "normal",
		bern_flag = False,
		filter_flag = True,
		neg_ent = 1,
		neg_rel = 0):
		
		self.in_path = in_path
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"
		"""set essential parameters"""
		self.work_threads = threads
		self.nbatches = nbatches
		self.batch_size = batch_size
		self.bern = bern_flag
		self.filter = filter_flag
		self.negative_ent = neg_ent
		self.negative_rel = neg_rel
		self.sampling_mode = sampling_mode
		self.cross_sampling_flag = 0
		# 读入数据
		self.read()

	def read(self):
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
		self.relTotal = base.getRelationTotal()
		self.entTotal = base.getEntityTotal()
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

	# 采样数据
	def sampling(self):
		base.sampling(
			self.batch_h,
			self.batch_t,
			self.batch_r,
			self.batch_y,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h, 
			"batch_t": self.batch_t, 
			"batch_r": self.batch_r, 
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	# 只替换 head 进行负采样, 生成数据
	def sampling_head(self):
		base.sampling(
			self.batch_h,
			self.batch_t,
			self.batch_r,
			self.batch_y,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	# 只替换 tail 进行负采样, 生成数据
	def sampling_tail(self):
		base.sampling(
			self.batch_h,
			self.batch_t,
			self.batch_r,
			self.batch_y,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	# 交替替换 head 和 tail 进行负采样, 生成数据
	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag 
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""interfaces to set essential parameters"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""interfaces to get essential parameters"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	# 迭代器
	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches