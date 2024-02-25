# coding:utf-8
#
# pybind11_ke/module/model/Model.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Feb 25, 2023
# 
# 该头文件定义了 Model.

"""Model 类 - 所有 KGE 模型的基类"""

import torch
from ..BaseModule import BaseModule

class Model(BaseModule):

	"""
	继承自 :py:class:`pybind11_ke.module.BaseModule`，仅仅增加了两个属性：:py:attr:`ent_tol` 和 :py:attr:`rel_tol`。
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int):

		"""创建 Model 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		"""

		super(Model, self).__init__()

		#: 实体的种类
		self.ent_tol: int = ent_tol
		#: 关系的种类
		self.rel_tol: int = rel_tol

	def forward(self) -> torch.Tensor:

		"""
		定义每次调用时执行的计算。该方法未实现，子类必须重写该方法，否则抛出 :py:class:`NotImplementedError` 错误。
		
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。

		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		raise NotImplementedError
	
	def predict(self) -> torch.Tensor:

		"""
		KGE 模型的推理方法。该方法未实现，子类必须重写该方法，否则抛出 :py:class:`NotImplementedError` 错误。
		
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		raise NotImplementedError

	def tri2emb(
		self,
		triples: torch.Tensor,
		negs: torch.Tensor = None,
		mode: str = 'single') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

		"""
		返回三元组对应的嵌入向量。
		
		:param triples: 正确的三元组
		:type triples: torch.Tensor
		:param negs: 负三元组类别
		:type negs: torch.Tensor
		:param mode: 模式
		:type triples: str
		:returns: 头实体、关系和尾实体的嵌入向量
		:rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
		"""
		
		if mode == "single":
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)
			
		elif mode == "head-batch" or mode == "head_predict":
			if negs is None:
				head_emb = self.ent_embeddings.weight.data.unsqueeze(0)
			else:
				head_emb = self.ent_embeddings(negs)
				
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)
			tail_emb = self.ent_embeddings(triples[:, 2]).unsqueeze(1)
			
		elif mode == "tail-batch" or mode == "tail_predict": 
			head_emb = self.ent_embeddings(triples[:, 0]).unsqueeze(1)
			relation_emb = self.rel_embeddings(triples[:, 1]).unsqueeze(1)
			
			if negs is None:
				tail_emb = self.ent_embeddings.weight.data.unsqueeze(0)
			else:
				tail_emb = self.ent_embeddings(negs)
		
		return head_emb, relation_emb, tail_emb