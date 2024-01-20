# coding:utf-8
#
# pybind11_ke/data/CompGCNTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
#
# 该脚本定义了 CompGCNTestSampler 类.

"""
CompGCNTestSampler - CompGCN 的测试数据采样器。
"""

from .CompGCNSampler import CompGCNSampler
from .GraphTestSampler import GraphTestSampler

class CompGCNTestSampler(GraphTestSampler):

    """``CompGCN`` :cite:`CompGCN` 的测试数据采样器。
    """

    def __init__(
        self,
        sampler: CompGCNSampler):

        """创建 CompGCNTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: CompGCNSampler
        """

        super().__init__(
            sampler=sampler
        )

        #: 训练集三元组
        self.triples: list[tuple[int, int, int]] = sampler.t_triples
        #: 幂
        self.power: float = -0.5