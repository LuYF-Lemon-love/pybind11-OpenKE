# coding:utf-8
#
# pybind11_ke/utils/EarlyStopping.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 5, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 5, 2024
#
# 该脚本定义了 EarlyStopping 类.

"""
EarlyStopping - 使用早停止避免过拟合。
"""

import os
import numpy as np
from ..module.model import Model

class EarlyStopping:

    """
    如果验证得分（越大越好）在给定的耐心后没有改善，则提前停止训练。
    """

    def __init__(
        self,
        save_path: str,
        patience: int = 2,
        verbose: bool = True,
        delta: float = 0):

        """创建 EarlyStopping 对象。

		:param save_path: 模型保存目录
		:type save_path: str
		:param patience: 上次验证得分改善后等待多长时间。默认值：2
		:type patience: int
		:param verbose: 如果为 True，则为每个验证得分改进打印一条消息。默认值：True
		:type verbose: bool
        :param delta: 监测数量的最小变化才符合改进条件。默认值：0
		:type delta: float
		"""
        
        #: 模型保存目录
        self.save_path: str = save_path
        #: 上次验证得分改善后等待多长时间。默认值：7
        self.patience: int = patience
        #: 如果为 True，则为每个验证得分改进打印一条消息。默认值：True
        self.verbose: bool = verbose
        #: 监测数量的最小变化才符合改进条件。默认值：0
        self.delta = delta
        
        #: 计数变量
        self.counter = 0
        #: 保存最好的得分
        self.best_score: float = -np.Inf
        #: 早停开关
        self.early_stop: bool = False

    def __call__(
        self,
        score: float,
        model: Model):

        """
		进行早停记录。
		"""
       
        if score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(
        self,
        score: float,
        model: Model):

        """
        当验证得分改善时保存模型。
        """
        
        if self.verbose:
            print(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        model.save_checkpoint(path)
        self.best_score = score