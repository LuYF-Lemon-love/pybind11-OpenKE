# coding:utf-8
#
# pybind11_ke/module/model/RotatE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 11, 2023
# 
# 该头文件定义了 RotatE.

"""
RotatE - 将实体表示成复数向量，关系建模为复数向量空间的旋转。
"""

import torch
import typing
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class RotatE(Model):

	"""
	``RotatE`` :cite:`RotatE` 提出于 2019 年，将实体表示成复数向量，关系建模为复数向量空间的旋转。

	评分函数为:

	.. math::
	
		\gamma - \parallel \mathbf{h} \circ \mathbf{r} - \mathbf{t} \parallel_{L_2}
	
	:math:`\circ` 表示哈达玛积（Hadamard product），正三元组的评分函数的值越大越好，如果想获得更详细的信息请访问 :ref:`RotatE <rotate>`。

	例子::

		from pybind11_ke.config import Trainer, Tester
		from pybind11_ke.module.model import RotatE
		from pybind11_ke.module.loss import SigmoidLoss
		from pybind11_ke.module.strategy import NegativeSampling

		# define the model
		rotate = RotatE(
			ent_tol = train_dataloader.get_ent_tol(),
			rel_tol = train_dataloader.get_rel_tol(),
			dim = 1024,
			margin = 6.0,
			epsilon = 2.0,
		)

		# define the loss function
		model = NegativeSampling(
			model = rotate, 
			loss = SigmoidLoss(adv_temperature = 2),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 0.0,
		)

		# test the model
		tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader, epochs = 6000,
			lr = 2e-5, opt_method = 'adam', use_gpu = True, device = 'cuda:1',
			tester = tester, test = True, valid_interval = 100,
			log_interval = 100, save_interval = 100,
			save_path = '../../checkpoint/rotate.pth', use_wandb = False)
		trainer.run()
	"""

	def __init__(
		self,
		ent_tol: int,
		rel_tol: int,
		dim: int = 100,
		margin: float = 6.0,
		epsilon: float = 2.0):

		"""创建 RotatE 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		:param margin: 原论文中损失函数的 gamma。
		:type margin: float
		:param epsilon: RotatE 原论文对应的源代码固定为 2.0。
		:type epsilon: float
		"""

		super(RotatE, self).__init__(ent_tol, rel_tol)

		#: RotatE 原论文对应的源代码固定为 2.0。
		self.epsilon: int = epsilon

		#: RotatE 原论文的实现中将实体嵌入向量的维度指定为 ``dim`` 的 2 倍。
		#: 因为实体嵌入向量需要划分为实部和虚部。
		self.dim_e: int = dim * 2
		#: 关系嵌入向量的维度，为 ``dim``。
		self.dim_r: int = dim

		#: 根据实体个数，创建的实体嵌入。
		self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim_e)
		#: 根据关系个数，创建的关系嵌入。
		self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim_r)

		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		#: 原论文中损失函数的 gamma。
		self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

	@override
	def forward(
		self,
		triples: torch.Tensor,
		negs: torch.Tensor = None,
		mode: str = 'single') -> torch.Tensor:

		"""
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param triples: 正确的三元组
		:type triples: torch.Tensor
		:param negs: 负三元组类别
		:type negs: torch.Tensor
		:param mode: 模式
		:type triples: str
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
		score = self.margin - self._calc(head_emb, relation_emb, tail_emb)
		return score

	def _calc(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor) -> torch.Tensor:

		"""计算 RotatE 的评分函数。

		利用 :py:func:`torch.chunk` 拆分实体嵌入向量获得复数的实部和虚部。
		原论文使用 L1-norm 作为距离函数，而这里使用的 L2-norm 作为距离函数。
		
		:param h: 头实体的向量。
		:type h: torch.Tensor
		:param r: 关系的向量。
		:type r: torch.Tensor
		:param t: 尾实体的向量。
		:type t: torch.Tensor
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		pi = self.pi_const

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		# Make phases of relations uniformly distributed in [-pi, pi]
		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)
	
		re_score = re_head * re_relation - im_head * im_relation
		im_score = re_head * im_relation + im_head * re_relation
		re_score = re_score - re_tail
		im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score

	@override
	def predict(
		self,
		data: dict[str, typing.Union[torch.Tensor,str]],
		mode) -> torch.Tensor:
		
		"""RotatE 的推理方法。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor,str]]
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

		triples = data["positive_sample"]
		head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
		score = self.margin - self._calc(head_emb, relation_emb, tail_emb)
		return score

	def regularization(
		self,
		data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

		"""L2 正则化函数（又称权重衰减），在损失函数中用到。
		
		:param data: 数据。
		:type data: dict[str, typing.Union[torch.Tensor, str]]
		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

		pos_sample = data["positive_sample"]
		neg_sample = data["negative_sample"]
		mode = data["mode"]
		pos_head_emb, pos_relation_emb, pos_tail_emb = self.tri2emb(pos_sample)
		neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)

		pos_regul = (torch.mean(pos_head_emb ** 2) + 
					 torch.mean(pos_relation_emb ** 2) + 
					 torch.mean(pos_tail_emb ** 2)) / 3

		neg_regul = (torch.mean(neg_head_emb ** 2) + 
					 torch.mean(neg_relation_emb ** 2) + 
					 torch.mean(neg_tail_emb ** 2)) / 3

		regul = (pos_regul + neg_regul) / 2

		return regul

def get_rotate_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`RotatE` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'model': {
				'value': 'RotatE'
			},
			'dim': {
				'values': [256, 512, 1024]
			},
			'margin': {
				'values': [1.0, 3.0, 6.0]
			},
			'epsilon': {
				'value': 2.0
			}
		}

	:returns: :py:class:`RotatE` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'RotatE'
		},
		'dim': {
			'values': [256, 512, 1024]
		},
		'margin': {
			'values': [1.0, 3.0, 6.0]
		},
		'epsilon': {
			'value': 2.0
		}
	}
		
	return parameters_dict