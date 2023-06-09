# coding:utf-8
#
# pybind11_ke/config/Tester.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 7, 2023
#
# 该脚本定义了验证模型类.

"""
Tester - 验证模型类，内部使用 ``tqmn`` 实现进度条。

基本用法如下：

.. code-block:: python

    from pybind11_ke.config import Trainer, Tester
    
    # test the model
    transe.load_checkpoint('../checkpoint/transe.ckpt')
    tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction()
"""

import torch
import numpy as np
from tqdm import tqdm
import base

class Tester(object):

    """
    :py:class:`Tester` 主要用于 KGE 模型的验证。
    """

    def __init__(self, model = None, data_loader = None, sampling_mode = 'link_test', use_gpu = True, device = "cuda:0"):

        """创建 Tester 对象。
        
        :param model: KGE 模型
        :type model: :py:class:`pybind11_ke.module.model.Model`
        :param data_loader: TestDataLoader
        :type data_loader: :py:class:`pybind11_ke.data.TestDataLoader`
        :param sampling_mode: :py:class:`pybind11_ke.data.TestDataLoader` 负采样的方式：``link_test`` or ``link_valid``
        :type sampling_mode: str
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :param device: 使用哪个 gpu
        :type device: str
        """

        #: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
        self.model = model
        #: :py:class:`pybind11_ke.data.TestDataLoader`
        self.data_loader = data_loader
        #: :py:class:`pybind11_ke.data.TestDataLoader` 负采样的方式：``link_test`` or ``link_valid``
        self.sampling_mode = sampling_mode
        #: 是否使用 gpu
        self.use_gpu = use_gpu
        #: gpu，利用 ``device`` 构造的 :py:class:`torch.device` 对象
        self.device = torch.device(device)
        
        if self.use_gpu:
            self.model.cuda(device = self.device)

    def to_var(self, x, use_gpu):

        """根据 ``use_gpu`` 返回 ``x`` 的张量
        
        :param x: 数据
        :type x: numpy.ndarray
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        :returns: 张量
        :rtype: torch.Tensor
        """

        if use_gpu:
            return torch.from_numpy(x).to(self.device)
        else:
            return torch.from_numpy(x)

    def test_one_step(self, data):

        """根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将模型验证 1 步。
        
        :param data: :py:attr:`data_loader` 利用
                        :py:meth:`pybind11_ke.data.TestDataLoader.sampling_lp` 函数生成的数据
        :type data: dict
        :returns: 三元组的得分
        :rtype: numpy.ndarray
        """
                
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self):
        
        """进行链接预测。

        :returns: 经典指标分别为 MRR，MR，Hits@1，Hits@3，Hits@10
        :rtype: tuple
        """

        base.init_test()
        self.data_loader.set_sampling_mode(self.sampling_mode)
        training_range = tqdm(self.data_loader)
        for [data_head, data_tail] in training_range:
            score = self.test_one_step(data_head)
            base.test_head(score, self.data_loader.type_constrain, self.sampling_mode)
            score = self.test_one_step(data_tail)
            base.test_tail(score, self.data_loader.type_constrain, self.sampling_mode)
        base.test_link_prediction(self.data_loader.type_constrain, self.sampling_mode)

        mrr = base.get_test_link_MRR(self.data_loader.type_constrain)
        mr = base.get_test_link_MR(self.data_loader.type_constrain)
        hit1 = base.get_test_link_Hit1(self.data_loader.type_constrain)
        hit3 = base.get_test_link_Hit3(self.data_loader.type_constrain)
        hit10 = base.get_test_link_Hit10(self.data_loader.type_constrain)
        return mrr, mr, hit1, hit3, hit10
    
    def set_sampling_mode(self, sampling_mode):
        
        """设置 :py:attr:`sampling_mode`
        
        :param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样，``tc`` 表示为分类进行负采样
        :type sampling_mode: str
        """
        
        self.sampling_mode = sampling_mode

    # def get_best_threshlod(self, score, ans):
    #     res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
    #     order = np.argsort(score)
    #     res = res[order]

    #     total_all = (float)(len(score))
    #     total_current = 0.0
    #     total_true = np.sum(ans)
    #     total_false = total_all - total_true

    #     res_mx = 0.0
    #     threshlod = None
    #     for index, [ans, score] in enumerate(res):
    #         if ans == 1:
    #             total_current += 1.0
    #         res_current = (2 * total_current + total_false - index - 1) / total_all
    #         if res_current > res_mx:
    #             res_mx = res_current
    #             threshlod = score
    #     return threshlod, res_mx

    # def run_triple_classification(self, threshlod = None):
    #     self.lib.init_test()
    #     self.data_loader.set_sampling_mode('classification')
    #     score = []
    #     ans = []
    #     training_range = tqdm(self.data_loader)
    #     for index, [pos_ins, neg_ins] in enumerate(training_range):
    #         res_pos = self.test_one_step(pos_ins)
    #         ans = ans + [1 for i in range(len(res_pos))]
    #         score.append(res_pos)

    #         res_neg = self.test_one_step(neg_ins)
    #         ans = ans + [0 for i in range(len(res_pos))]
    #         score.append(res_neg)

    #     score = np.concatenate(score, axis = -1)
    #     ans = np.array(ans)

    #     if threshlod == None:
    #         threshlod, _ = self.get_best_threshlod(score, ans)

    #     res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
    #     order = np.argsort(score)
    #     res = res[order]

    #     total_all = (float)(len(score))
    #     total_current = 0.0
    #     total_true = np.sum(ans)
    #     total_false = total_all - total_true

    #     for index, [ans, score] in enumerate(res):
    #         if score > threshlod:
    #             acc = (2 * total_current + total_false - index) / total_all
    #             break
    #         elif ans == 1:
    #             total_current += 1.0

    #     return acc, threshlod