# coding:utf-8
#
# pybind11_ke/module/model/Analogy.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on June 17, 2023
# 
# 该头文件定义了 Analogy.

"""
Analogy - DistMult、HolE 和 ComplEx 的集大成者，效果与 HolE、ComplEx 差不多。

论文地址: `Analogical Inference for Multi-relational Embeddings <https://proceedings.mlr.press/v70/liu17d.html>`__ 。

基本用法如下：

.. code-block:: python

	from pybind11_ke.config import Trainer, Tester
	from pybind11_ke.module.model import Analogy
	from pybind11_ke.module.loss import SoftplusLoss
	from pybind11_ke.module.strategy import NegativeSampling

	# define the model
	analogy = Analogy(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 200
	)

	# define the loss function
	model = NegativeSampling(
		model = analogy, 
		loss = SoftplusLoss(),
		batch_size = train_dataloader.get_batch_size(), 
		regul_rate = 1.0
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader,
	                  train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	analogy.save_checkpoint('../checkpoint/analogy.ckpt')

	# test the model
	analogy.load_checkpoint('../checkpoint/analogy.ckpt')
	tester = Tester(model = analogy, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
"""

import torch
import torch.nn as nn
from .Model import Model

class Analogy(Model):

	"""
	Analogy 类，继承自 :py:class:`pybind11_ke.module.model.Model`。
	
	Analogy 提出于 2017 年，:py:class:`pybind11_ke.module.model.DistMult`、
		:py:class:`pybind11_ke.module.model.HolE` 和 :py:class:`pybind11_ke.module.model.ComplEx` 的集大成者，
		效果与 :py:class:`pybind11_ke.module.model.HolE`、:py:class:`pybind11_ke.module.model.ComplEx` 差不多。 

	评分函数为: :math:`<\operatorname{Re}(\mathbf{r}), \operatorname{Re}(\mathbf{h}), \operatorname{Re}(\mathbf{t})> + <\operatorname{Re}(\mathbf{r}), \operatorname{Im}(\mathbf{h}), \operatorname{Im}(\mathbf{t})> + <\operatorname{Im}(\mathbf{r}), \operatorname{Re}(\mathbf{h}), \operatorname{Im}(\mathbf{t})> - <\operatorname{Im}(\mathbf{r}), \operatorname{Im}(\mathbf{h}), \operatorname{Re}(\mathbf{t})>`，
        :math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >` 为逐元素多线性点积（element-wise multi-linear dot product），
	正三元组的评分函数的值越大越好，负三元组越小越好。
	"""

	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(Analogy, self).__init__(ent_tot, rel_tot)

		self.dim = dim
		self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim * 2)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim * 2)
		
		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def _calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
		return (-torch.sum(r_re * h_re * t_re +
						   r_re * h_im * t_im +
						   r_im * h_re * t_im -
						   r_im * h_im * t_re, -1)
				-torch.sum(h * t * r, -1))

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		h = self.ent_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		t = self.ent_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h_re = self.ent_re_embeddings(batch_h)
		h_im = self.ent_im_embeddings(batch_h)
		h = self.ent_embeddings(batch_h)
		t_re = self.ent_re_embeddings(batch_t)
		t_im = self.ent_im_embeddings(batch_t)
		t = self.ent_embeddings(batch_t)
		r_re = self.rel_re_embeddings(batch_r)
		r_im = self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h_re ** 2) + 
				 torch.mean(h_im ** 2) + 
				 torch.mean(h ** 2) + 
				 torch.mean(t_re ** 2) + 
				 torch.mean(t_im ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r_re ** 2) + 
				 torch.mean(r_im ** 2) + 
				 torch.mean(r ** 2)) / 9
		return regul

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()