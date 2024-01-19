# coding:utf-8
#
# pybind11_ke/data/CompGCNTestSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
#
# 该脚本定义了 CompGCNTestSampler 类.

from .GraphTestSampler import GraphTestSampler

class CompGCNTestSampler(GraphTestSampler):

    """Sampling graph for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
        triples: The training triples.
    """

    def __init__(self, sampler):

        super().__init__(
            sampler=sampler
        )
        self.triples = sampler.t_triples
        #: 幂
        self.power = -0.5