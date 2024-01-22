# coding:utf-8
#
# pybind11_ke/module/model/__init__.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 28, 2023
# 
# 该头文件定义了 model 接口.

"""KGE 模型部分。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Model import Model
from .TransE import TransE, get_transe_hpo_config
from .TransH import TransH, get_transh_hpo_config
from .TransR import TransR, get_transr_hpo_config
from .TransD import TransD, get_transd_hpo_config
from .RotatE import RotatE, get_rotate_hpo_config
from .DistMult import DistMult
from .ComplEx import ComplEx
from .RESCAL import RESCAL
from .Analogy import Analogy
from .SimplE import SimplE
from .HolE import HolE
from .RGCN import RGCN
from .CompGCN import CompGCN, CompGCNCov

__all__ = [
    'Model',
    'TransE',
    'get_transe_hpo_config',
    'TransH',
    'get_transh_hpo_config',
    'TransR',
    'get_transr_hpo_config',
    'TransD',
    'get_transd_hpo_config',
    'RotatE',
    'get_rotate_hpo_config',
    'DistMult',
    'ComplEx',
    'RESCAL',
    'Analogy',
    'SimplE',
    'HolE',
    'RGCN',
    'CompGCN',
    'CompGCNCov'
]