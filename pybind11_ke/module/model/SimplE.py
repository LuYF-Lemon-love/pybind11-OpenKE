# coding:utf-8
#
# pybind11_ke/module/model/SimplE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 7, 2023
# 
# 该头文件定义了 SimplE.

"""
SimplE - 简单的双线性模型，能够为头实体和尾实体学习不同的嵌入向量。
"""

import math
import torch
import typing
import numpy as np
import torch.nn as nn
from .Model import Model
from typing_extensions import override

class SimplE(Model):
    
    """
    ``SimplE`` :cite:`SimplE` 提出于 2018 年，简单的双线性模型，能够为头实体和尾实体学习不同的嵌入向量。
    
    评分函数为: 

    .. math::
    
        1/2(<\mathbf{h}_{i}, \mathbf{v}_r, \mathbf{t}_{j}> + <\mathbf{h}_{j}, \mathbf{v}_{r^{-1}}, \mathbf{t}_{i}>)
    
    :math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >` 为逐元素多线性点积（element-wise multi-linear dot product）。

    正三元组的评分函数的值越大越好，负三元组越小越好，如果想获得更详细的信息请访问 :ref:`SimplE <simple>`。

    例子::

        from pybind11_ke.config import Trainer, Tester
        from pybind11_ke.module.model import SimplE
        from pybind11_ke.module.loss import SoftplusLoss
        from pybind11_ke.module.strategy import NegativeSampling

        # define the model
        simple = SimplE(
        	ent_tol = train_dataloader.get_ent_tol(),
        	rel_tol = train_dataloader.get_rel_tol(),
        	dim = config.dim
        )

        # define the loss function
        model = NegativeSampling(
        	model = simple, 
        	loss = SoftplusLoss(),
        	batch_size = train_dataloader.get_batch_size(), 
        	regul_rate = config.regul_rate
        )

        # dataloader for test
        test_dataloader = TestDataLoader(in_path = config.in_path)

        # test the model
        tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)

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

        """创建 SimplE 对象。
        
        :param ent_tol: 实体的个数
        :type ent_tol: int
        :param rel_tol: 关系的个数
        :type rel_tol: int
        :param dim: 实体嵌入向量和关系嵌入向量的维度
        :type dim: int
        """
        
        super(SimplE, self).__init__(ent_tol, rel_tol)
        
        #: 实体嵌入向量和关系嵌入向量的维度
        self.dim: int = dim

        #: 根据实体个数，创建的实体嵌入
        self.ent_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim * 2)
        #: 根据关系个数，创建的关系嵌入
        self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim * 2)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        
        """
        定义每次调用时执行的计算。
        :py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        利用 :py:func:`torch.clamp` 裁剪最后的的得分，防止遇到 NaN 问题。
        """

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
        
        """计算 SimplE 的评分函数。
        
        :param h: 头实体的向量。
        :type h: torch.Tensor
        :param r: 关系的向量。
        :type r: torch.Tensor
        :param t: 尾实体的向量。
        :type t: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor
        """

        hh_embs, th_embs = torch.chunk(h, 2, dim=-1)
        r_embs, r_inv_embs = torch.chunk(r, 2, dim=-1)
        ht_embs, tt_embs = torch.chunk(t, 2, dim=-1)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, -1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, -1)
        
        # Without clipping, we run into NaN problems.
        # 基于论文作者的实现。
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    @override
    def predict(
        self,
        data: dict[str, typing.Union[torch.Tensor,str]],
        mode) -> torch.Tensor:
        
        """SimplE 的推理方法。
        
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

def get_simple_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`SimplE` 的默认超参数优化配置。
	
	默认配置为::
	
	    parameters_dict = {
	    	'model': {
	    		'value': 'SimplE'
	    	},
	    	'dim': {
	    		'values': [50, 100, 200]
	    	}
	    }

	:returns: :py:class:`SimplE` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'SimplE'
		},
		'dim': {
			'values': [50, 100, 200]
		}
	}
		
	return parameters_dict