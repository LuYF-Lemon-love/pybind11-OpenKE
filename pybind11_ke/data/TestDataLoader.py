# coding:utf-8
#
# pybind11_ke/data/TestDataLoader.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
#
# 该脚本定义了采样数据的函数, 用于验证模型.

"""
TrainDataLoader - 数据集类，类似 :py:class:`torch.utils.data.DataLoader`。
"""

import base
import torch
import typing
import numpy as np
from collections.abc import Callable

class TestDataSampler(object):

	"""将 :py:meth:`pybind11_ke.data.TestDataLoader.sampling` 进行封装。
	"""

	def __init__(
		self,
		data_total: int,
		sampler: Callable[[], dict[str, typing.Union[torch.Tensor, str]]]):

		"""创建 TestDataSampler 对象

		:param data_total: 测试集多少个三元组 
		:type data_total: int
		:param sampler: 采样器
		:type sampler: :py:meth:`pybind11_ke.data.TestDataLoader.sampling`
		"""

		#: 测试集多少个三元组
		self.data_total: int = data_total
		#: :py:meth:`pybind11_ke.data.TestDataLoader.sampling` 函数
		self.sampler: Callable[[], dict[str, typing.Union[torch.Tensor, str]]] = sampler
		self.total: int = 0

	def __iter__(self):

		"""迭代器函数 :py:meth:`iterator.__iter__`"""

		return self

	def __next__(self) -> dict[str, typing.Union[torch.Tensor, str]]:

		"""
		迭代器函数 :py:meth:`iterator.__next__`
		
		:returns: 采样一批数据
		:rtype: dict[str, typing.Union[torch.Tensor, str]]
		"""

		self.total += 1 
		if self.total > self.data_total:
			raise StopIteration()
		return self.sampler()

	def __len__(self) -> int:

		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`data_total`
		:rtype: int
		"""

		return self.data_total

class TestDataLoader(object):

	"""
	主要从底层 C++ 模块获得数据用于 KGE 模型的评估。
	
	例子::

		from pybind11_ke.config import Tester
		from pybind11_ke.data import TestDataLoader

		# dataloader for test
		test_dataloader = TestDataLoader('../../benchmarks/FB15K/')

		# test the model
		tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')
	"""

	def __init__(
		self,
		in_path: str = "./",
		ent_file: str = "entity2id.txt",
		rel_file: str = "relation2id.txt",
		train_file: str = "train2id.txt",
		valid_file: str = "valid2id.txt",
		test_file: str = "test2id.txt",
		sampling_mode: str = 'link_test',
		type_constrain: bool = True):

		"""创建 TestDataLoader 对象。

		:param in_path: 数据集目录
		:type in_path: str
		:param ent_file: entity2id.txt
		:type ent_file: str
		:param rel_file: relation2id.txt
		:type rel_file: str
		:param train_file: train2id.txt
		:type train_file: str
		:param valid_file: valid2id.txt
		:type valid_file: str
		:param test_file: test2id.txt
		:type test_file: str
		:param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
		:type sampling_mode: str
		:param type_constrain: 是否用 type_constrain.txt 进行负采样
		:type type_constrain: bool
		"""

		#: 数据集目录
		self.in_path: str = in_path
		#: entity2id.txt
		self.ent_file: str = ent_file
		#: relation2id.txt
		self.rel_file: str = rel_file
		#: train2id.txt
		self.train_file: str = train_file
		#: valid2id.txt
		self.valid_file: str = valid_file
		#: test2id.txt
		self.test_file: str = test_file
		#: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
		self.sampling_mode: str = sampling_mode
		#: 是否用 type_constrain.txt 进行负采样
		self.type_constrain: bool = type_constrain

		#: 实体的个数
		self.ent_tol: int = 0
		#: 关系的个数
		self.rel_tol: int = 0
		#: 测试集三元组的个数
		self.test_tol: int = 0
		#: 验证集三元组的个数
		self.valid_tol: int = 0

		# 读入数据
		self.read()

	def read(self):

		"""利用 `pybind11 <https://github.com/pybind/pybind11>`__ 让底层 C++ 模块读取数据集中的数据"""

		print("Start reading validation and testing data...")

		base.set_in_path(self.in_path)
		base.set_ent_path(self.ent_file)
		base.set_rel_path(self.rel_file)
		base.set_train_path(self.train_file)
		base.set_valid_path(self.valid_file)
		base.set_test_path(self.test_file)
		base.rand_reset()
		base.read_test_files()

		if self.type_constrain:
			base.read_type_files()

		print("Validation and testing data read completed.\n")

		self.ent_tol = base.get_entity_total()
		self.rel_tol = base.get_relation_total()
		self.test_tol = base.get_test_total()
		self.valid_tol = base.get_valid_total()

		# 利用 np.zeros 分配内存
		self.test_h = np.zeros(self.ent_tol, dtype=np.int64)
		self.test_t = np.zeros(self.ent_tol, dtype=np.int64)
		self.test_r = np.zeros(self.ent_tol, dtype=np.int64)

		self.test_pos_h = np.zeros(self.test_tol, dtype=np.int64)
		self.test_pos_t = np.zeros(self.test_tol, dtype=np.int64)
		self.test_pos_r = np.zeros(self.test_tol, dtype=np.int64)
		
		self.test_neg_h = np.zeros(self.test_tol, dtype=np.int64)
		self.test_neg_t = np.zeros(self.test_tol, dtype=np.int64)
		self.test_neg_r = np.zeros(self.test_tol, dtype=np.int64)

	# 为链接预测进行采样数据
	def sampling(self) -> dict[str, typing.Union[torch.Tensor, str]]:

		"""为链接预测进行采样数据，为给定的正三元组，用所有实体依次替换头尾实体得到
		2 * :py:attr:`ent_tol` 个三元组。
		
		:returns: 对于一个正三元组生成的所有可能破化的三元组
		:rtype: dict[str, typing.Union[torch.Tensor, str]]
		"""

		res = []
		base.get_head_batch(self.test_h, self.test_t, self.test_r, self.sampling_mode)
		res.append({
			"batch_h": self.test_h.copy(), 
			"batch_t": self.test_t[:1].copy(),
			"batch_r": self.test_r[:1].copy(),
			"mode": "head_batch"
		})
		base.get_tail_batch(self.test_h, self.test_t, self.test_r, self.sampling_mode)
		res.append({
			"batch_h": self.test_h[:1],
			"batch_t": self.test_t,
			"batch_r": self.test_r[:1],
			"mode": "tail_batch"
		})
		return res

	def get_ent_tol(self) -> int:

		"""返回 :py:attr:`ent_tol`

		:returns: :py:attr:`ent_tol`
		:rtype: int
		"""
		
		return self.ent_tol

	def get_rel_tol(self) -> int:

		"""返回 :py:attr:`rel_tol`

		:returns: :py:attr:`rel_tol`
		:rtype: int
		"""

		return self.rel_tol

	def get_test_tol(self) -> int:

		"""返回 :py:attr:`test_tol`

		:returns: :py:attr:`test_tol`
		:rtype: int
		"""

		return self.test_tol

	def get_valid_tol(self) -> int:

		"""返回 :py:attr:`test_tol`

		:returns: :py:attr:`test_tol`
		:rtype: int
		"""

		return self.valid_tol

	def set_sampling_mode(self, sampling_mode: str):

		"""设置 :py:attr:`sampling_mode`
		
		:param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
		:type sampling_mode: str
		"""

		self.sampling_mode = sampling_mode

	def __iter__(self) -> TestDataSampler:

		"""迭代器函数 :py:meth:`iterator.__iter__`，
		根据 :py:attr:`sampling_mode` 决定是评估验证集还是测试集。"""

		if self.sampling_mode == "link_test":
			base.init_test()
			return TestDataSampler(self.test_tol, self.sampling)
		elif self.sampling_mode == "link_valid":
			base.init_test()
			return TestDataSampler(self.valid_tol, self.sampling)
		else:
			raise ValueError("pybind11_ke.data.TestDataLoader.sampling can only be a link_test or link_valid.")

	def __len__(self) -> int:
		
		"""len() 要求 :py:meth:`object.__len__`
		
		:returns: :py:attr:`test_tol` 或 :py:attr:`valid_tol`
		:rtype: int
		"""
		
		if self.sampling_mode == "link_test":
			return self.test_tol
		elif self.sampling_mode == "link_valid":
			return self.valid_tol

def get_test_data_loader_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`TestDataLoader` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'valid_file': {
				'value': 'valid2id.txt'
			},
			'test_file': {
				'value': 'test2id.txt'
			},
			'type_constrain': {
				'value': True
			},
			'neg_ent': {
				'values': [1, 4, 16, 64]
			},
			'neg_rel': {
				'value': 0
			}
		}

	:returns: :py:class:`TestDataLoader` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'valid_file': {
			'value': 'valid2id.txt'
		},
		'test_file': {
			'value': 'test2id.txt'
		},
		'type_constrain': {
			'value': True
		},
		'neg_ent': {
			'values': [1, 4, 16, 64]
		},
		'neg_rel': {
			'value': 0
		}
	}
		
	return parameters_dict