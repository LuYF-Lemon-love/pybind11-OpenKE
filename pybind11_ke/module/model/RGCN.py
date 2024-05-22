# coding:utf-8
#
# pybind11_ke/module/model/RGCN.py
# 
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 22, 2024
# 
# 该头文件定义了 R-GCN.

"""
R-GCN - 第一个图神经网络模型。
"""

import dgl
import torch
import typing
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from dgl.nn.pytorch import RelGraphConv
from typing_extensions import override

class RGCN(Model):
    
    """
	``R-GCN`` :cite:`R-GCN` 提出于 2017 年，是第一个图神经网络模型。
	
	正三元组的评分函数的值越大越好，如果想获得更详细的信息请访问 :ref:`R-GCN <rgcn>`。

    例子::

        from pybind11_ke.data import GraphDataLoader
        from pybind11_ke.module.model import RGCN
        from pybind11_ke.module.loss import RGCNLoss
        from pybind11_ke.module.strategy import RGCNSampling
        from pybind11_ke.config import Trainer, GraphTester
        
        dataloader = GraphDataLoader(
        	in_path = "../../benchmarks/FB15K237/",
        	batch_size = 60000,
        	neg_ent = 10,
        	test = True,
        	test_batch_size = 100,
        	num_workers = 16
        )
        
        # define the model
        rgcn = RGCN(
        	ent_tol = dataloader.get_ent_tol(),
        	rel_tol = dataloader.get_rel_tol(),
        	dim = 500,
        	num_layers = 2
        )
        
        # define the loss function
        model = RGCNSampling(
        	model = rgcn,
        	loss = RGCNLoss(model = rgcn, regularization = 1e-5)
        )
        
        # test the model
        tester = GraphTester(model = rgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0')
        
        # train the model
        trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
        	epochs = 10000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
        	tester = tester, test = True, valid_interval = 500, log_interval = 500,
        	save_interval = 500, save_path = '../../checkpoint/rgcn.pth'
        )
        trainer.run()
	"""

    def __init__(
        self,
        ent_tol: int,
        rel_tol: int,
        dim: int,
        num_layers: int):

        """创建 RGCN 对象。

		:param ent_tol: 实体的个数
		:type ent_tol: int
		:param rel_tol: 关系的个数
		:type rel_tol: int
		:param dim: 实体和关系嵌入向量的维度
		:type dim: int
		:param num_layers: 图神经网络的层数
		:type num_layers: int
		"""

        super(RGCN, self).__init__(ent_tol, rel_tol)

        #: 实体和关系嵌入向量的维度
        self.dim: int = dim
        #: 图神经网络的层数
        self.num_layers: int = num_layers

        #: 根据实体个数，创建的实体嵌入
        self.ent_emb: torch.nn.Embedding = None
        #: 根据关系个数，创建的关系嵌入
        self.rel_emb: torch.nn.parameter.Parameter = None
        #: R-GCN 的图神经网络层
        self.RGCN: torch.nn.ModuleList = None
        #: 图神经网络层的输出
        self.Loss_emb: torch.nn.Embedding = None

        self.build_model()

    def build_model(self):

        """构建模型"""

        self.ent_emb = nn.Embedding(self.ent_tol, self.dim)

        self.rel_emb = nn.Parameter(torch.Tensor(self.rel_tol, self.dim))

        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.RGCN = nn.ModuleList()
        for idx in range(self.num_layers):
            RGCN_idx = self.build_hidden_layer(idx)
            self.RGCN.append(RGCN_idx)

    def build_hidden_layer(
        self,
        idx: int) -> dgl.nn.pytorch.conv.RelGraphConv:

        """返回第 idx 的图神经网络层。
        
        :param idx: 数据。
        :type idx: int
        :returns: 图神经网络层
        :rtype: dgl.nn.pytorch.conv.RelGraphConv
        """

        act = F.relu if idx < self.num_layers - 1 else None
        return RelGraphConv(self.dim, self.dim, self.rel_tol, "bdd",
                    num_bases=100, activation=act, self_loop=True, dropout=0.2)

    @override  
    def forward(
        self,
        graph: dgl.DGLGraph,
        ent: torch.Tensor,
        rel: torch.Tensor,
        norm: torch.Tensor,
        triples: torch.Tensor,
        mode: str = 'single') -> torch.Tensor:

        """
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        
        :param graph: 子图
        :type graph: dgl.DGLGraph
        :param ent: 子图的实体
        :type ent: torch.Tensor
        :param rel: 子图的关系
        :type rel: torch.Tensor
        :param norm: 关系的归一化系数
        :type norm: torch.Tensor
        :param triples: 三元组
        :type triples: torch.Tensor
        :param mode: 模式
        :type mode: str
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""
        
        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = self.distmult_score_func(head_emb, rela_emb, tail_emb, mode)

        return score

    def tri2emb(
        self,
        embedding: torch.Tensor,
        triples: torch.Tensor,
        mode: str = "single") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
		获得三元组对应头实体、关系和尾实体的嵌入向量。
        
        :param embedding: 经过图神经网络更新的实体嵌入向量
        :type embedding: torch.Tensor
        :param triples: 训练的三元组
        :type triples: torch.Tensor
        :param mode: 模式
        :type mode: str
        :returns: 头实体、关系和尾实体的嵌入向量
        :rtype: torch.Tensor
		"""

        rela_emb = self.rel_emb[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
        head_emb = embedding[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim] 
        tail_emb = embedding[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]

        if mode == "head-batch" or mode == "head_predict":
            head_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        elif mode == "tail-batch" or mode == "tail_predict":
            tail_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        return head_emb, rela_emb, tail_emb

    def distmult_score_func(
        self,
        head_emb: torch.Tensor,
        relation_emb: torch.Tensor,
        tail_emb: torch.Tensor,
        mode: str) -> torch.Tensor:

        """
		计算 DistMult 的评分函数。
        
        :param head_emb: 头实体嵌入向量
        :type head_emb: torch.Tensor
        :param relation_emb: 关系嵌入向量
        :type relation_emb: torch.Tensor
        :param tail_emb: 尾实体嵌入向量
        :type tail_emb: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""

        if mode == 'head-batch':
            score = head_emb * (relation_emb * tail_emb)
        else:
            score = (head_emb * relation_emb) * tail_emb

        score = score.sum(dim = -1)
        return score

    @override
    def predict(
        self,
        data: dict[str, torch.Tensor],
        mode: str) -> torch.Tensor:

        """R-GCN 的推理方法。
        
        :param data: 数据。
        :type data: dict[str, torch.Tensor]
        :param mode: 模式
        :type mode: str
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""

        triples    = data['positive_sample']
        graph      = data['graph']
        ent        = data['entity']
        rel        = data['rela']
        norm       = data['norm']

        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = self.distmult_score_func(head_emb, rela_emb, tail_emb, mode)

        return score

def get_rgcn_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`RGCN` 的默认超参数优化配置。
	
	默认配置为::
	
	    parameters_dict = {
	    	'model': {
	    		'value': 'RGCN'
	    	},
	    	'dim': {
	    		'values': [200, 300, 400]
	    	},
	    	'num_layers': {
	    		'value': 2
	    	}
	    }

	:returns: :py:class:`RGCN` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]
	"""

	parameters_dict = {
		'model': {
			'value': 'RGCN'
		},
		'dim': {
			'values': [200, 300, 400]
		},
		'num_layers': {
			'value': 2
		}
	}
		
	return parameters_dict