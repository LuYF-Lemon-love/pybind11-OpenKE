# coding:utf-8
#
# pybind11_ke/module/model/Model.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
# 
# 该头文件定义了 Model.

"""Model - 所有 KGE 模型的基类"""

from ..BaseModule import BaseModule

class Model(BaseModule):
	"""Model 继承自 :py:class:`pybind11_ke.module.BaseModule`，
	仅仅增加了两个属性：:py:attr:`ent_tot` 和 :py:attr:`rel_tot`。"""

	def __init__(self, ent_tot, rel_tot):
		"""创建 Model 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		"""

		super(Model, self).__init__()

		#: 实体的种类
		self.ent_tot = ent_tot
		#: 关系的种类
		self.rel_tot = rel_tot

	def forward(self):
		"""定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。"""

		raise NotImplementedError
	
	def predict(self):
		"""KGE 模型的推理方法。"""

		raise NotImplementedError