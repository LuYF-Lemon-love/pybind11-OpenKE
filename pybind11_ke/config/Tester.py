# coding:utf-8
#
# pybind11_ke/config/Tester.py
#
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 3, 2023
#
# 该脚本定义了验证模型类.

"""
Tester - 验证模型类，内部使用 ``tqmn`` 实现进度条。
"""

import base
import torch
from tqdm import tqdm

class Tester(object):

    """
    主要用于 KGE 模型的评估。

    例子::

        from pybind11_ke.config import Trainer, Tester

        # test the model
        transe.load_checkpoint('../checkpoint/transe.ckpt')
        tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
        tester.run_link_prediction()
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
                        :py:meth:`pybind11_ke.data.TestDataLoader.sampling` 函数生成的数据
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

        :returns: 经典指标分别为 MR，MRR，Hits@1，Hits@3，Hits@10
        :rtype: tuple
        """

        self.data_loader.set_sampling_mode(self.sampling_mode)
        training_range = tqdm(self.data_loader)
        self.model.eval()
        with torch.no_grad():
            for [data_head, data_tail] in training_range:
                score = self.test_one_step(data_head)
                base.test_head(score, self.data_loader.type_constrain, self.sampling_mode)
                score = self.test_one_step(data_tail)
                base.test_tail(score, self.data_loader.type_constrain, self.sampling_mode)
        base.test_link_prediction(self.data_loader.type_constrain, self.sampling_mode)

        mr = base.get_test_link_MR()
        mrr = base.get_test_link_MRR()
        hit1 = base.get_test_link_Hit1()
        hit3 = base.get_test_link_Hit3()
        hit10 = base.get_test_link_Hit10()

        if self.data_loader.type_constrain:
            mrTC = base.get_test_link_MR(True)
            mrrTC = base.get_test_link_MRR(True)
            hit1TC = base.get_test_link_Hit1(True)
            hit3TC = base.get_test_link_Hit3(True)
            hit10TC = base.get_test_link_Hit10(True)

            return mr, mrr, hit1, hit3, hit10, mrTC, mrrTC, hit1TC, hit3TC, hit10TC, 
        
        return mr, mrr, hit1, hit3, hit10
    
    def set_sampling_mode(self, sampling_mode):
        
        """设置 :py:attr:`sampling_mode`
        
        :param sampling_mode: 数据采样模式，``link_test`` 和 ``link_valid`` 分别表示为链接预测进行测试集和验证集的负采样
        :type sampling_mode: str
        """
        
        self.sampling_mode = sampling_mode

def get_tester_hpo_config():

	"""返回 :py:class:`Tester` 的默认超参数优化配置。"""

	parameters_dict = {
        'use_gpu': {
			'value': True
		},
		'device': {
			'value': 'cuda:0'
		},
	}
		
	return parameters_dict