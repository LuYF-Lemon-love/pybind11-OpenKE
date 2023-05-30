# coding:utf-8
#
# pybind11_ke/config/Tester.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 30, 2023
#
# 该脚本定义了验证模型类.

"""
Tester - 验证模型类，内部使用 ``tqmn`` 实现进度条。

基本用法如下：

.. code-block:: python

    from pybind11_ke.config import Trainer, Tester
    
    # test the model
    transe.load_checkpoint('./checkpoint/transe.ckpt')
    tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
"""

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import base

class Tester(object):

    """
	Tester 主要用于 KGE 模型的验证。
	"""

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        """创建 Tester 对象。

		:param model: KGE 模型
		:type model: :py:class:`pybind11_ke.module.model.Model`
		:param data_loader: TestDataLoader
		:type data_loader: :py:class:`pybind11_ke.data.TestDataLoader`
		:param use_gpu: 是否使用 gpu
		:type use_gpu: bool
		"""

        #: KGE 模型，即 :py:class:`pybind11_ke.module.model.Model`
        self.model = model
        #: :py:class:`pybind11_ke.data.TestDataLoader`
        self.data_loader = data_loader
        #: 是否使用 gpu
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        """设置 :py:attr:`model`
        
        :param model: KGE 模型
        :type model: :py:class:`pybind11_ke.module.model.Model`
        """

        self.model = model

    def set_data_loader(self, data_loader):
        """设置 :py:attr:`data_loader`
        
        :param data_loader: TestDataLoader
        :type data_loader: :py:class:`pybind11_ke.data.TestDataLoader`
        """

        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        """设置 :py:attr:`use_gpu`
        
        :param use_gpu: 是否使用 gpu
        :type use_gpu: bool
        """

        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

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
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        """根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将
		模型验证 1 步。

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

    def run_link_prediction(self, type_constrain = False):
        """进行链接预测
        
        :param type_constrain: 是否用 type_constrain.txt 进行负采样
        :type type_constrain: bool
        :returns: 经典指标分别为 MRR，MR，Hits@10，Hits@3，Hits@1
		:rtype: tuple
        """

        base.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            base.testHead(score, index, type_constrain)
            score = self.test_one_step(data_tail)
            base.testTail(score, index, type_constrain)
        base.test_link_prediction(type_constrain)

        mrr = base.getTestLinkMRR(type_constrain)
        mr = base.getTestLinkMR(type_constrain)
        hit10 = base.getTestLinkHit10(type_constrain)
        hit3 = base.getTestLinkHit3(type_constrain)
        hit1 = base.getTestLinkHit1(type_constrain)
        # print(hit10)
        return mrr, mr, hit10, hit3, hit1

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
    #     self.lib.initTest()
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