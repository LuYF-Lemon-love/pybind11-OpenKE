# coding:utf-8
#
# pybind11_ke/module/model/ComplEx.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 15, 2023
# 
# 该头文件定义了 ComplEx.

"""
ComplEx - 第一个真正意义上复数域模型，简单而且高效。

论文地址: `Complex Embeddings for Simple Link Prediction <https://arxiv.org/abs/1606.06357>`__ 。

基本用法如下：

.. code-block:: python

    from pybind11_ke.config import Trainer, Tester
    from pybind11_ke.module.model import ComplEx
    from pybind11_ke.module.loss import SoftplusLoss
    from pybind11_ke.module.strategy import NegativeSampling

    # define the model
    complEx = ComplEx(
    	ent_tot = train_dataloader.get_ent_tot(),
    	rel_tot = train_dataloader.get_rel_tot(),
    	dim = 200
    )

    # define the loss function
    model = NegativeSampling(
    	model = complEx, 
    	loss = SoftplusLoss(),
    	batch_size = train_dataloader.get_batch_size(), 
    	regul_rate = 1.0
    )

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader,
    				train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
    trainer.run()
    complEx.save_checkpoint('../checkpoint/complEx.ckpt')

    # test the model
    complEx.load_checkpoint('../checkpoint/complEx.ckpt')
    tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class ComplEx(Model):

    """
	ComplEx 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	ComplEx 提出于 2016 年，第一个真正意义上复数域模型，简单而且高效。复数版本的 
        :py:class:`pybind11_ke.module.model.DistMult`。

	评分函数为: :math:`<\operatorname{Re}(\mathbf{r}), \operatorname{Re}(\mathbf{h}), \operatorname{Re}(\mathbf{t})> + <\operatorname{Re}(\mathbf{r}), \operatorname{Im}(\mathbf{h}), \operatorname{Im}(\mathbf{t})> + <\operatorname{Im}(\mathbf{r}), \operatorname{Re}(\mathbf{h}), \operatorname{Im}(\mathbf{t})> - <\operatorname{Im}(\mathbf{r}), \operatorname{Im}(\mathbf{h}), \operatorname{Re}(\mathbf{t})>`
    
    其中 :math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >` 为逐元素多线性点积（element-wise multi-linear dot product），
	正三元组的评分函数的值越大越好，负三元组越小越好。
	"""

    def __init__(self, ent_tot, rel_tot, dim = 100):

        """创建 ComplEx 对象。

		:param ent_tot: 实体的个数
		:type ent_tot: int
		:param rel_tot: 关系的个数
		:type rel_tot: int
		:param dim: 实体嵌入向量和关系嵌入的维度
		:type dim: int
		"""

        super(ComplEx, self).__init__(ent_tot, rel_tot)

        #: 实体嵌入向量和关系嵌入向量的维度
        self.dim = dim
        #: 根据实体个数，创建的实体嵌入的实部
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        #: 根据实体个数，创建的实体嵌入的虚部
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        #: 根据关系个数，创建的关系嵌入的实部
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        #: 根据关系个数，创建的关系嵌入的虚部
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):

        """计算 ComplEx 的评分函数。
		
        :param h_re: 头实体的实部向量。
        :type h_re: torch.Tensor
        :param h_im: 头实体的虚部向量。
        :type h_im: torch.Tensor
        :param t_re: 尾实体的实部向量。
        :type t_re: torch.Tensor
        :param t_im: 尾实体的虚部向量。
        :type t_im: torch.Tensor
        :param r_re: 关系的实部向量。
        :type r_re: torch.Tensor
        :param r_im: 关系的虚部向量。
        :type r_im: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""

        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

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
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
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
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        regul = (torch.mean(h_re ** 2) + 
                 torch.mean(h_im ** 2) + 
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def predict(self, data):

        """ComplEx 的推理方法。
		
		:param data: 数据。
		:type data: dict
		:returns: 三元组的得分
		:rtype: numpy.ndarray
		"""

        score = -self.forward(data)
        return score.cpu().data.numpy()