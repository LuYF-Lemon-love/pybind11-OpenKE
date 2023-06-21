# coding:utf-8
#
# pybind11_ke/module/model/SimplE.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 21, 2023
# 
# 该头文件定义了 SimplE.

"""
SimplE - 简单的双线性模型，能够为头实体和尾实体学习不同的嵌入向量。

论文地址: `SimplE Embedding for Link Prediction in Knowledge Graphs <https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html>`__ 。

基本用法如下：

.. code-block:: python

    from pybind11_ke.config import Trainer, Tester
    from pybind11_ke.module.model import SimplE
    from pybind11_ke.module.loss import SoftplusLoss
    from pybind11_ke.module.strategy import NegativeSampling

    # define the model
    simple = SimplE(
    	ent_tot = train_dataloader.get_ent_tot(),
    	rel_tot = train_dataloader.get_rel_tot(),
    	dim = 200
    )

    # define the loss function
    model = NegativeSampling(
    	model = simple, 
    	loss = SoftplusLoss(),
    	batch_size = train_dataloader.get_batch_size(), 
    	regul_rate = 1.0
    )

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader,
                      train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
    trainer.run()
    simple.save_checkpoint('../checkpoint/simple.ckpt')

    # test the model
    simple.load_checkpoint('../checkpoint/simple.ckpt')
    tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class SimplE(Model):

    """
	SimplE 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	SimplE 提出于 2018 年，简单的双线性模型，能够为头实体和尾实体学习不同的嵌入向量。

	评分函数为: :math:`(<\mathbf{h}_{i}, \mathbf{v}_r, \mathbf{t}_{j}> + <\mathbf{h}_{j}, \mathbf{v}_{r^{-1}}, \mathbf{t}_{i}>)`，
        :math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >` 为逐元素多线性点积（element-wise multi-linear dot product），
	正三元组的评分函数的值越大越好，负三元组越小越好。:math:`X_{0:5} =\frac{1}{r^2} (X_0, X_1, X_2, X_3, X_4)`
	"""

    def __init__(self, ent_tot, rel_tot, dim = 100):

        """创建 SimplE 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体嵌入向量和关系嵌入向量的维度
		:type dim: int
		"""

        super(SimplE, self).__init__(ent_tot, rel_tot)

        #: 实体嵌入向量和关系嵌入向量的维度
        self.dim = dim
        #: 根据实体个数，创建的实体嵌入
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        #: 根据关系个数，创建的关系嵌入
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        #: 根据关系个数，创建的逆关系嵌入
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    def _calc_avg(self, h, t, r, r_inv):
        return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def forward(self, data):
        
        """
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: torch.Tensor
		"""

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        score = self._calc_avg(h, t, r, r_inv)
        return score

    def regularization(self, data):

        """L2 正则化函数（又称权重衰减），在损失函数中用到。
		
		:param data: 数据。
		:type data: dict
		:returns: 模型参数的正则损失
		:rtype: torch.Tensor
		"""

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 4
        return regul

    def predict(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = -self._calc_ingr(h, r, t)
        return score.cpu().data.numpy()