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
from .RGCNTestSampler import RGCNTestSampler

class CompGCNTestSampler(RGCNTestSampler):

    """``CompGCN`` :cite:`CompGCN` 的测试数据采样器。

    例子::

        from pybind11_ke.data import RGCNTestSampler, CompGCNTestSampler
        from torch.utils.data import DataLoader

        #: 测试数据采样器
        test_sampler: typing.Union[typing.Type[RGCNTestSampler], typing.Type[CompGCNTestSampler]] = test_sampler(
            sampler=train_sampler,
            valid_file=valid_file,
            test_file=test_file,
        )
    
        #: 验证集三元组
        data_val: list[tuple[int, int, int]] = test_sampler.get_valid()
        #: 测试集三元组
        data_test: list[tuple[int, int, int]] = test_sampler.get_test()

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
        test_file: str = "test2id.txt",
        type_constrain: bool = True):

        """创建 CompGCNTestSampler 对象。

        :param sampler: 训练数据采样器。
        :type sampler: CompGCNSampler
        :param valid_file: valid2id.txt
        :type valid_file: str
        :param test_file: test2id.txt
        :type test_file: str
        :param type_constrain: 是否报告 type_constrain.txt 限制的测试结果
        :type type_constrain: bool
        """

        super().__init__(
            sampler=sampler,
            valid_file = valid_file,
            test_file = test_file,
            type_constrain = type_constrain
        )
        
        #: 幂
        self.power: float = -0.5