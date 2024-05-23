# coding:utf-8
#
# pybind11_ke/module/loss/CompGCNLoss.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2024
#
# 该脚本定义了 CompGCNLoss 类.

"""
CompGCNLoss - 损失函数类，CompGCN 原论文中应用这种损失函数完成模型学习。
"""

import torch
from .Loss import Loss
from typing import Any
from ..model import CompGCN

class CompGCNLoss(Loss):
    
    """
    ``CompGCN`` :cite:`CompGCN` 原论文中应用这种损失函数完成模型训练。
    
    .. Note:: :py:meth:`forward` 中的正样本评分函数的得分应大于负样本评分函数的得分。
    
    例子::

        from pybind11_ke.module.loss import CompGCNLoss
        from pybind11_ke.module.strategy import CompGCNSampling

        # define the loss function
        model = CompGCNSampling(
            model = compgcn,
            loss = CompGCNLoss(model = compgcn),
            ent_tol = dataloader.get_ent_tol()
        )
    """

    def __init__(
        self,
        model: CompGCN):

        """创建 CompGCNLoss 对象。
        
        :param model: 模型
        :type model: CompGCN
		"""

        super(CompGCNLoss, self).__init__()

        #: 模型
        self.model: CompGCN = model
        #: 损失函数
        self.loss: torch.nn.BCELoss = torch.nn.BCELoss()

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> torch.Tensor:

        """计算 CompGCNLoss 损失函数。定义每次调用时执行的计算。:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        
        :param pred: 模型的得分。
        :type pred: torch.Tensor
        :param labels: 标签
        :type labels: torch.Tensor
        :returns: 损失值
        :rtype: torch.Tensor
		"""
        
        loss = self.loss(pred, label)
        return loss

def get_compgcn_loss_hpo_config() -> dict[str, dict[str, Any]]:

	"""返回 :py:class:`CompGCNLoss` 的默认超参数优化配置。
	
	默认配置为::
	
	    parameters_dict = {
	    	'loss': {
	    		'value': 'CompGCNLoss'
	    	}
	    }
	
	:returns: :py:class:`CompGCNLoss` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'loss': {
			'value': 'CompGCNLoss'
		}
	}
		
	return parameters_dict