# coding:utf-8
#
# pybind11_ke/module/loss/Cross_Entropy_Loss.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 20, 2024
#
# 该脚本定义了 Cross_Entropy_Loss 类.

"""
Cross_Entropy_Loss - 损失函数类，CompGCN 原论文中应用这种损失函数完成模型学习。
"""

import torch
from .Loss import Loss
from ..model import CompGCN

class Cross_Entropy_Loss(Loss):

    """
	``CompGCN`` :cite:`CompGCN` 原论文中应用这种损失函数完成模型训练。
	
	.. Note:: :py:meth:`forward` 中的正样本评分函数的得分应大于负样本评分函数的得分。

	例子::

        from pybind11_ke.module.loss import Cross_Entropy_Loss
        from pybind11_ke.module.strategy import CompGCNSampling
        
        # define the loss function
        model = CompGCNSampling(
        	model = compgcn,
        	loss = Cross_Entropy_Loss(model = compgcn),
        	ent_tol = dataloader.train_sampler.ent_tol
        )
	"""

    def __init__(
        self,
        model: CompGCN):

        """创建 Cross_Entropy_Loss 对象。
        
        :param model: 模型
        :type model: CompGCN
		"""

        super(Cross_Entropy_Loss, self).__init__()

        #: 模型
        self.model: CompGCN = model
        #: 损失函数
        self.loss: torch.nn.BCELoss = torch.nn.BCELoss()

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> torch.Tensor:

        """计算 Cross_Entropy_Loss 损失函数。定义每次调用时执行的计算。:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        
        :param pred: 模型的得分。
        :type pred: torch.Tensor
        :param labels: 标签
        :type labels: torch.Tensor
        :returns: 损失值
        :rtype: torch.Tensor
		"""
        
        loss = self.loss(pred, label)
        return loss