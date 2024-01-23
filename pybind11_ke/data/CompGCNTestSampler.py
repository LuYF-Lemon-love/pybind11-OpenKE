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

    例子::

        from pybind11_ke.data import CompGCNTestSampler
        from torch.utils.data import DataLoader

        #: 测试数据采样器
        test_sampler: typing.Type[CompGCNTestSampler] = CompGCNTestSampler(
            sampler=train_sampler
        )

        val_dataloader = DataLoader(
            data_val,
            shuffle=False,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=test_sampler.sampling,
        )

        test_dataloader = DataLoader(
            data_test,
            shuffle=False,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=test_sampler.sampling,
        )
    """

    def __init__(
        self,
        sampler: CompGCNSampler,
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt"):

        """创建 CompGCNTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: CompGCNSampler
        """

        super().__init__(
            sampler=sampler,
            valid_file = valid_file,
            test_file = test_file
        )

        #: 训练集三元组
        self.triples: list[tuple[int, int, int]] = sampler.t_triples
        #: 幂
        self.power: float = -0.5