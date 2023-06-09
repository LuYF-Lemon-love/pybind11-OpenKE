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
from .MarginLoss import MarginLoss
from .SoftplusLoss import SoftplusLoss
from .SigmoidLoss import SigmoidLoss

__all__ = [
    'Loss',
    'MarginLoss',
    'SoftplusLoss',
    'SigmoidLoss',
]