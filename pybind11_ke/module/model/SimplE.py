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
        #: 根据实体个数，创建的头实体嵌入
        self.ent_h_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
        #: 根据实体个数，创建的尾实体嵌入
        self.ent_t_embeddings: torch.nn.Embedding = nn.Embedding(self.ent_tol, self.dim)
        #: 根据关系个数，创建的关系嵌入
        self.rel_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)
        #: 根据关系个数，创建的逆关系嵌入
        self.rel_inv_embeddings: torch.nn.Embedding = nn.Embedding(self.rel_tol, self.dim)

        sqrt_size = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_h_embeddings.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embeddings.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embeddings.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embeddings.weight.data, -sqrt_size, sqrt_size)

    @override   
    def forward(
        self,
        data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:
        
        """
        定义每次调用时执行的计算。
        :py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        利用 :py:func:`torch.clamp` 裁剪最后的的得分，防止遇到 NaN 问题。
        
        :param data: 数据。
        :type data: dict[str, typing.Union[torch.Tensor, str]]
        :returns: 三元组的得分
        :rtype: torch.Tensor
        """

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        hh_embs = self.ent_h_embeddings(batch_h)
        ht_embs = self.ent_h_embeddings(batch_t)
        th_embs = self.ent_t_embeddings(batch_h)
        tt_embs = self.ent_t_embeddings(batch_t)
        r_embs = self.rel_embeddings(batch_r)
        r_inv_embs = self.rel_inv_embeddings(batch_r)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, -1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, -1)

        # Without clipping, we run into NaN problems.
        # 基于论文作者的实现。
        return torch.clamp((scores1 + scores2) / 2, -20, 20)
        
    def regularization(
        self,
        data: dict[str, typing.Union[torch.Tensor, str]]) -> torch.Tensor:

        """L2 正则化函数（又称权重衰减），在损失函数中用到。
        
        :param data: 数据。
        :type data: dict[str, typing.Union[torch.Tensor, str]]
        :returns: 模型参数的正则损失
        :rtype: torch.Tensor
        """

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        hh_embs = self.ent_h_embeddings(batch_h)
        ht_embs = self.ent_h_embeddings(batch_t)
        th_embs = self.ent_t_embeddings(batch_h)
        tt_embs = self.ent_t_embeddings(batch_t)
        r_embs = self.rel_embeddings(batch_r)
        r_inv_embs = self.rel_inv_embeddings(batch_r)

        regul = (torch.mean(hh_embs ** 2) + 
                 torch.mean(ht_embs ** 2) + 
                 torch.mean(th_embs ** 2) +
                 torch.mean(tt_embs ** 2) +
                 torch.mean(r_embs ** 2) +
                 torch.mean(r_inv_embs ** 2)) / 6

        return regul

    @override
    def predict(
        self,
        data: dict[str, typing.Union[torch.Tensor,str]]) -> np.ndarray:

        """SimplE 的推理方法。
        
        :param data: 数据。
        :type data: dict[str, typing.Union[torch.Tensor,str]]
        :returns: 三元组的得分
        :rtype: numpy.ndarray
        """
        
        score = -self.forward(data)
        return score.cpu().data.numpy()