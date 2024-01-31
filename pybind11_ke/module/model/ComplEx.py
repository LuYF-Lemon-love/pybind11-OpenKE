# coding:utf-8
#
# pybind11_ke/module/model/ComplEx.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 6, 2023
# 
# 该头文件定义了 ComplEx.

"""
ComplEx - 第一个真正意义上复数域模型，简单而且高效。
"""

import torch
import typing
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class ComplEx(Model):

    """
    ``ComplEx`` :cite:`ComplEx` 提出于 2016 年，第一个真正意义上复数域模型，简单而且高效。复数版本的 :py:class:`pybind11_ke.module.model.DistMult`。

    评分函数为:

    .. math::

        <\operatorname{Re}(h),\operatorname{Re}(r),\operatorname{Re}(t)>
            +<\operatorname{Re}(h),\operatorname{Im}(r),\operatorname{Im}(t)>
            +<\operatorname{Im}(h),\operatorname{Re}(r),\operatorname{Im}(t)>
            -<\operatorname{Im}(h),\operatorname{Im}(r),\operatorname{Re}(t)>

    :math:`h, r, t \in \mathbb{C}^n` 是复数向量，:math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >=\sum_{i=1}^{n}a_ib_ic_i` 为逐元素多线性点积（element-wise multi-linear dot product）。
	
    正三元组的评分函数的值越大越好，负三元组越小越好，如果想获得更详细的信息请访问 :ref:`ComplEx <complex>`。

    例子::

        from pybind11_ke.config import Trainer, Tester
        from pybind11_ke.module.model import ComplEx
        from pybind11_ke.module.loss import SoftplusLoss
        from pybind11_ke.module.strategy import NegativeSampling

        # define the model
        complEx = ComplEx(
        	ent_tol = train_dataloader.get_ent_tol(),
        	rel_tol = train_dataloader.get_rel_tol(),
        	dim = config.dim
        )

        # define the loss function
        model = NegativeSampling(
        	model = complEx, 
        	loss = SoftplusLoss(),
        	batch_size = train_dataloader.get_batch_size(), 
        	regul_rate = config.regul_rate
        )

        # test the model
        tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)

        # train the model
        trainer = Trainer(model = model, data_loader = train_dataloader, epochs = config.epochs,
        	lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
        	tester = tester, test = config.test, valid_interval = config.valid_interval,
        	log_interval = config.log_interval, save_interval = config.save_interval,
        	save_path = config.save_path, use_wandb = True)
        trainer.run()
	"""

    def __init__(
        self,
        ent_tol: int,
        rel_tol: int,
        dim: int = 100):

        """创建 ComplEx 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体嵌入向量和关系嵌入向量的维度
		:type dim: int
		"""

        super(ComplEx, self).__init__(ent_tol, rel_tol)

        #: 实体嵌入向量和关系嵌入向量的维度
        self.dim: int = dim
        #: 根据实体个数，创建的实体嵌入
        self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim * 2)
        #: 根据关系个数，创建的关系嵌入
        self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim * 2)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
   
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
        score = self._calc(head_emb, relation_emb, tail_emb)
        return score

    def _calc(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor) -> torch.Tensor:
        
        """计算 ComplEx 的评分函数。
        
        :param h: 头实体的向量。
        :type h: torch.Tensor
        :param r: 关系的向量。
        :type r: torch.Tensor
        :param t: 尾实体的向量。
        :type t: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor
        """

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_relation, im_relation = torch.chunk(r, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        
        return torch.sum(
            re_head * re_tail * re_relation
            + im_head * im_tail * re_relation
            + re_head * im_tail * im_relation
            - im_head * re_tail * im_relation,
            -1
        )
        
    @override
    def predict(
        self,
        data: dict[str, typing.Union[torch.Tensor,str]],
        mode) -> torch.Tensor:
        
        """ComplEx 的推理方法。
        
        :param data: 数据。
        :type data: dict[str, typing.Union[torch.Tensor,str]]
        :returns: 三元组的得分
        :rtype: torch.Tensor
        """

        triples = data["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self._calc(head_emb, relation_emb, tail_emb)
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
        if mode == "bern":
            neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(neg_sample)
        else:
            neg_head_emb, neg_relation_emb, neg_tail_emb = self.tri2emb(pos_sample, neg_sample, mode)
            
        pos_regul = (torch.mean(pos_head_emb ** 2) + 
                     torch.mean(pos_relation_emb ** 2) + 
                     torch.mean(pos_tail_emb ** 2)) / 3
                     
        neg_regul = (torch.mean(neg_head_emb ** 2) + 
                     torch.mean(neg_relation_emb ** 2) + 
                     torch.mean(neg_tail_emb ** 2)) / 3
                     
        regul = (pos_regul + neg_regul) / 2

        return regul

def get_complex_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`ComplEx` 的默认超参数优化配置。
	
	默认配置为::
	
	    parameters_dict = {
	    	'model': {
	    		'value': 'ComplEx'
	    	},
	    	'dim': {
	    		'values': [50, 100, 200]
	    	}
	    }

	:returns: :py:class:`ComplEx` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'ComplEx'
		},
		'dim': {
			'values': [50, 100, 200]
		}
	}
		
	return parameters_dict
