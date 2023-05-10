# coding:utf-8
#
# pybind11-ke/data/__init__.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 10, 2023
#
# 该脚本定义了接口.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .TrainDataLoader import TrainDataLoader
from .TestDataLoader import TestDataLoader

__all__ = [
	'TrainDataLoader',
	'TestDataLoader'
]