# coding:utf-8
#
# pybind11_ke/module/loss/RGCN_Loss.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 17, 2023
#
# 该脚本定义了 RGCNLoss 类.

"""
RGCNLoss - 损失函数类，R-GCN 原论文中应用这种损失函数完成模型学习。
"""

import torch
from ..model import RGCN
import torch.nn.functional as F
from .Loss import Loss

class RGCNLoss(Loss):
    
    """
	``R-GCN`` :cite:`R-GCN` 原论文中应用这种损失函数完成模型训练。
	
	.. Note:: :py:meth:`forward` 中的正样本评分函数的得分应大于负样本评分函数的得分。

	例子::

        from pybind11_ke.module.loss import RGCNLoss
        from pybind11_ke.module.strategy import RGCNSampling
                
        # define the loss function
        model = RGCNSampling(
        	model = rgcn,
        	loss = RGCNLoss(model = rgcn, regularization = 1e-5)
        )
	"""

    def __init__(
        self,
        model: RGCN,
        regularization: float):

        """创建 RGCNLoss 对象。
        
        :param model: 模型
        :type model: RGCN
        :param regularization: 正则率
        :type regularization: float
		"""
        
        super(RGCNLoss, self).__init__()

        #: 模型
        self.model: RGCN = model
        #: 正则率
        self.regularization: float = regularization
    
    def reg_loss(self) -> torch.Tensor:

        """获得正则部分的损失。
        
        :returns: 损失值
        :rtype: torch.Tensor
		"""

        return torch.mean(self.model.Loss_emb.pow(2)) + torch.mean(self.model.rel_emb.pow(2))

    def forward(
        self,
        score: torch.Tensor,
        labels: torch.Tensor) -> torch.Tensor:

        """计算 RGCNLoss 损失函数。定义每次调用时执行的计算。:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        
        :param score: 模型的得分。
        :type score: torch.Tensor
        :param labels: 标签
        :type labels: torch.Tensor
        :returns: 损失值
        :rtype: torch.Tensor
		"""

        loss = F.binary_cross_entropy_with_logits(score, labels)
        regu = self.regularization * self.reg_loss()
        loss += regu
        return loss