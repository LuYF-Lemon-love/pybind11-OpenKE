# coding:utf-8
#
# pybind11_ke/config/TrainerAccelerator.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Apr 12, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Apr 12, 2024
#
# 该脚本定义了并行训练循环函数.

"""
利用 accelerate 实现并行训练。
"""

from typing import Any, List
from accelerate import Accelerator

def accelerator_prepare(*args: List[Any]) -> List[Any]:

	"""
	由于分布式并行依赖于 `accelerate <https://github.com/huggingface/accelerate>`_ ，因此，需要利用 Accelerator 为分布式训练准备对象。

	:param args: :py:class:`pybind11_ke.data.KGEDataLoader` 和 :py:class:`pybind11_ke.module.strategy.Strategy` 。
	:type args: typing.List[typing.Any]
	:returns: 包装好的对象。
	:rtype: typing.List[typing.Any]
	"""
	
	accelerator = Accelerator()
	result = accelerator.prepare(*args)
	result = list(result)
	result.append(accelerator)
	return result