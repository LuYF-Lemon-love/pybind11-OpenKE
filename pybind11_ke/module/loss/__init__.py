# coding:utf-8
#
# pybind11_ke/module/loss/__init__.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
# 
# 该头文件定义了 loss 接口.

"""损失函数部分。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Loss import Loss
from .MarginLoss import MarginLoss, get_margin_loss_hpo_config
from .SigmoidLoss import SigmoidLoss, get_sigmoid_loss_hpo_config
from .SoftplusLoss import SoftplusLoss, get_softplus_loss_hpo_config
from .RGCNLoss import RGCNLoss, get_rgcn_loss_hpo_config
from .CompGCNLoss import CompGCNLoss, get_compgcn_loss_hpo_config

__all__ = [
    'Loss',
    'MarginLoss',
    'get_margin_loss_hpo_config',
    'SigmoidLoss',
    'get_sigmoid_loss_hpo_config',
    'SoftplusLoss',
    'get_softplus_loss_hpo_config',
    'RGCNLoss',
    'get_rgcn_loss_hpo_config',
    'CompGCNLoss',
    'get_compgcn_loss_hpo_config'
]