# coding:utf-8
#
# pybind11_ke/module/strategy/__init__.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
# 
# 该头文件定义了 strategy 接口.

"""训练策略部分。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Strategy import Strategy
from .NegativeSampling import NegativeSampling, get_negative_sampling_hpo_config
from .RGCNSampling import RGCNSampling, get_rgcn_sampling_hpo_config
from .CompGCNSampling import CompGCNSampling, get_compgcn_sampling_hpo_config
from .Sampling import Sampling

__all__ = [
    'Strategy',
    'NegativeSampling',
    'get_negative_sampling_hpo_config',
    'RGCNSampling',
    'get_rgcn_sampling_hpo_config',
    'CompGCNSampling',
    'get_compgcn_sampling_hpo_config',
    'Sampling'
]