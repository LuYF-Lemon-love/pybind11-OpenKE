# coding:utf-8
#
# pybind11_ke/module/loss/MarginLoss.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
#
# 该脚本定义了 margin-based ranking criterion 损失函数.

"""
MarginLoss - 损失函数类，TransE 原论文中应用这种损失函数完成模型学习。

基本用法如下：

.. code-block:: python

	from pybind11_ke.module.loss import MarginLoss
	from pybind11_ke.module.strategy import NegativeSampling
	
	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class MarginLoss(Loss):

	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			
	
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()