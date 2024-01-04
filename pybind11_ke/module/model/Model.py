# coding:utf-8
#
# pybind11_ke/module/model/Model.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
# 
# 该头文件定义了 Model.

"""Model 类 - 所有 KGE 模型的基类"""

import torch
import numpy as np
from ..BaseModule import BaseModule

class Model(BaseModule):

	"""
	继承自 :py:class:`pybind11_ke.module.BaseModule`，仅仅增加了两个属性：:py:attr:`ent_tot` 和 :py:attr:`rel_tot`。
	"""

	def __init__(
		self,
		ent_tot: int,
		rel_tot: int):

		"""创建 Model 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		"""

		super(Model, self).__init__()

		#: 实体的种类
		self.ent_tot: int = ent_tot
		#: 关系的种类
		self.rel_tot: int = rel_tot

	def forward(self) -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。

		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		raise NotImplementedError
	
	def predict(self) -> np.ndarray:

		"""
		KGE 模型的推理方法。
		
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

		raise NotImplementedError