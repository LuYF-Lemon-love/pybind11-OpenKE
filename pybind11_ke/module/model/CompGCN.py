# coding:utf-8
#
# pybind11_ke/module/model/CompGCN.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2024
#
# 该脚本定义了 CompGCN 类.

"""
CompGCN - 这是一种在图卷积网络中整合多关系信息的新框架，它利用知识图谱嵌入技术中的各种组合操作，将实体和关系共同嵌入到图中。
"""

import dgl
import torch
import typing
from torch import nn
import dgl.function as fn
from .Model import Model
import torch.nn.functional as F
from typing_extensions import override

class CompGCN(Model):

    """
    ``CompGCN`` :cite:`CompGCN` 发表于 ``2020`` 年，这是一种在图卷积网络中整合多关系信息的新框架，它利用知识图谱嵌入技术中的各种组合操作，将实体和关系共同嵌入到图中。

    正三元组的评分函数的值越大越好，如果想获得更详细的信息请访问 :ref:`CompGCN <compgcn>`。

    例子::

        from pybind11_ke.module.model import CompGCN
        from pybind11_ke.module.loss import CompGCNLoss
        from pybind11_ke.module.strategy import CompGCNSampling
        from pybind11_ke.config import Trainer, GraphTester
        
        # define the model
        compgcn = CompGCN(
        	ent_tol = dataloader.get_ent_tol(),
        	rel_tol = dataloader.get_rel_tol(),
        	dim = 100
        )
        
        # define the loss function
        model = CompGCNSampling(
        	model = compgcn,
        	loss = CompGCNLoss(model = compgcn),
        	ent_tol = dataloader.get_ent_tol()
        )
        
        # test the model
        tester = GraphTester(model = compgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")
        
        # train the model
        trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
        	epochs = 2000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
        	tester = tester, test = True, valid_interval = 50, log_interval = 50,
        	save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
        )
        trainer.run()
    """

    def __init__(
        self,
        ent_tol: int,
        rel_tol: int,
        dim: int,
        opn: str = 'mult',
        fet_drop: float = 0.2,
        hid_drop: float = 0.3,
        margin: float = 40.0,
        decoder_model: str = 'ConvE'):

        """创建 RGCN 对象。
        
        :param ent_tol: 实体的个数
        :type ent_tol: int
        :param rel_tol: 关系的个数
        :type rel_tol: int
        :param dim: 实体和关系嵌入向量的维度
        :type dim: int
        :param opn: 组成运算符：'mult'、'sub'、'corr'
        :type opn: str
        :param fet_drop: 用于 'ConvE' 解码器，用于卷积特征的 dropout
        :type fet_drop: float
        :param hid_drop: 用于 'ConvE' 解码器，用于隐藏层的 dropout
        :type hid_drop: float
        :param margin: 用于 'TransE' 解码器，gamma。
        :type margin: float
        :param decoder_model: 用什么得分函数作为解码器: 'ConvE'、'DistMult'、'TransE'
        :type decoder_model: str
		"""

        super(CompGCN, self).__init__(ent_tol, rel_tol)

        #: 实体和关系嵌入向量的维度
        self.dim: int = dim
        #: 组成运算符：'mult'、'sub'、'corr'
        self.opn: str = opn
        #: 用什么得分函数作为解码器: 'ConvE'、'DistMult'
        self.decoder_model: str = decoder_model

        #------------------------------CompGCN--------------------------------------------------------------------
        #: 根据实体个数，创建的实体嵌入
        self.ent_emb: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor(self.ent_tol, self.dim))
        #: 根据关系个数，创建的关系嵌入
        self.rel_emb: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor(self.rel_tol, self.dim))

        nn.init.xavier_normal_(self.ent_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        #: CompGCNCov
        self.GraphCov: CompGCNCov = CompGCNCov(self.dim, self.dim * 2, torch.tanh, bias = 'False', drop_rate = 0.1, opn = self.opn)
        #: 用于 :py:attr:`GraphCov` 输出结果
        self.drop: torch.nn.Dropout = nn.Dropout(0.3)
        #: 最后计算得分时的偏置
        self.bias: torch.nn.parameter.Parameter = nn.Parameter(torch.zeros(self.ent_tol))
        #-----------------------------ConvE-----------------------------------------------------------------------
        #: 用于 'ConvE' 解码器，头实体嵌入向量和关系嵌入向量的 BatchNorm
        self.bn0: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(1)
        #: 用于 'ConvE' 解码器，卷积层
        self.conv1: torch.nn.Conv2d = torch.nn.Conv2d(1, 200, (7, 7), 1, 0, bias=False)
        #: 用于 'ConvE' 解码器，卷积特征的 BatchNorm
        self.bn1: torch.nn.Conv2d = torch.nn.BatchNorm2d(200)
        #: 用于 'ConvE' 解码器，卷积特征的 Dropout
        self.fet_drop: torch.nn.Dropout = torch.nn.Dropout2d(fet_drop)
        flat_sz_h = 4 * self.dim // 20 - 7 + 1
        flat_sz_w = 20 - 7 + 1
        flat_sz = flat_sz_h * flat_sz_w * 200
        #: 用于 'ConvE' 解码器，隐藏层层
        self.fc: torch.nn.Linear = torch.nn.Linear(flat_sz, self.dim*2)
        #: 用于 'ConvE' 解码器，隐藏层的 Dropout
        self.hid_drop: torch.nn.Dropout = torch.nn.Dropout(hid_drop)
        #: 用于 'ConvE' 解码器，隐藏层的 BatchNorm
        self.bn2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self.dim*2)
        #-----------------------------TransE-----------------------------------------------------------------------
        #: 用于 TransE 得分函数
        self.margin: torch.nn.parameter.Parameter = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    @override
    def forward(
        self,
        graph: dgl.DGLGraph,
        relation: torch.Tensor,
        norm: torch.Tensor,
        triples: torch.Tensor) -> torch.Tensor:

        """
		定义每次调用时执行的计算。
		:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
        
        :param graph: 子图
        :type graph: dgl.DGLGraph
        :param relation: 子图的关系
        :type relation: torch.Tensor
        :param norm: 关系的归一化系数
        :type norm: torch.Tensor
        :param triples: 三元组
        :type triples: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""

        head, rela = triples[:,0], triples[:, 1]
        x, r = self.ent_emb, self.rel_emb
        x, r = self.GraphCov(graph, x, r, relation, norm)
        x = self.drop(x)
        head_emb = torch.index_select(x, 0, head)
        rela_emb = torch.index_select(r, 0, rela)

        if self.decoder_model.lower() == 'conve':
           score = self.conve(head_emb, rela_emb, x)
        elif self.decoder_model.lower() == 'distmult':
            score = self.distmult(head_emb, rela_emb, x)
        elif self.decoder_model.lower() == 'transe':
            score = self.transe(head_emb, rela_emb, x)
        else:
            raise ValueError("please choose decoder (TransE/DistMult/ConvE)")

        return score

    def conve(
        self,
        sub_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        all_ent: torch.Tensor) -> torch.Tensor:

        """计算 ConvE 作为解码器时三元组的得分。
        
        :param sub_emb: 头实体的嵌入向量
        :type sub_emb: torch.Tensor
        :param rel_emb: 关系的嵌入向量
        :type rel_emb: torch.Tensor
        :param all_ent: 全部实体的嵌入向量
        :type all_ent: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor"""

        stack_input = self.concat(sub_emb, rel_emb)
        x = self.bn0(stack_input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fet_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def concat(
        self,
        ent_embed: torch.Tensor,
        rel_embed: torch.Tensor) -> torch.Tensor:

        """ConvE 作为解码器时，用于拼接头实体嵌入向量和关系嵌入向量。
        
        :param ent_embed: 头实体的嵌入向量
        :type ent_embed: torch.Tensor
        :param rel_embed: 关系的嵌入向量
        :type rel_embed: torch.Tensor
        :returns: ConvE 解码器的输入特征
        :rtype: torch.Tensor"""

        ent_embed = ent_embed.view(-1, 1, self.dim*2)
        rel_embed = rel_embed.view(-1, 1, self.dim*2)
        stack_input = torch.cat([ent_embed, rel_embed], 1)
        stack_input = stack_input.reshape(-1, 1, 4*self.dim//20, 20)
        return stack_input

    def distmult(
        self,
        head_emb: torch.Tensor,
        rela_emb: torch.Tensor,
        all_ent: torch.Tensor) -> torch.Tensor:

        """计算 DistMult 作为解码器时三元组的得分。
        
        :param sub_emb: 头实体的嵌入向量
        :type sub_emb: torch.Tensor
        :param rel_emb: 关系的嵌入向量
        :type rel_emb: torch.Tensor
        :param all_ent: 全部实体的嵌入向量
        :type all_ent: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor"""

        obj_emb = head_emb * rela_emb
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def transe(
        self,
        head_emb: torch.Tensor,
        rela_emb: torch.Tensor,
        all_ent: torch.Tensor) -> torch.Tensor:

        """计算 TransE 作为解码器时三元组的得分。
        
        :param sub_emb: 头实体的嵌入向量
        :type sub_emb: torch.Tensor
        :param rel_emb: 关系的嵌入向量
        :type rel_emb: torch.Tensor
        :param all_ent: 全部实体的嵌入向量
        :type all_ent: torch.Tensor
        :returns: 三元组的得分
        :rtype: torch.Tensor"""

        obj_emb = head_emb + rela_emb
        x = self.margin - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)
        return score

    @override
    def predict(
        self,
        data: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]],
        mode: str) -> torch.Tensor:

        """CompGCN 的推理方法。
        
        :param data: 数据。
        :type data: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
        :param mode: 在 CompGCN 时，无用，只为了保证推理函数形式一致
        :type mode: str
        :returns: 三元组的得分
        :rtype: torch.Tensor
		"""
        
        triples    = data['positive_sample']
        graph      = data['graph']
        relation   = data['rela']
        norm       = data['norm'] 

        head, rela = triples[:,0], triples[:, 1]
        x, r = self.ent_emb, self.rel_emb
        x, r = self.GraphCov(graph, x, r, relation, norm)
        x = self.drop(x)
        head_emb = torch.index_select(x, 0, head)
        rela_emb = torch.index_select(r, 0, rela)

        if self.decoder_model.lower() == 'conve':
           score = self.conve(head_emb, rela_emb, x)
        elif self.decoder_model.lower() == 'distmult':
            score = self.distmult(head_emb, rela_emb, x)
        elif self.decoder_model.lower() == 'transe':
            score = self.transe(head_emb, rela_emb, x)
        else:
            raise ValueError("please choose decoder (TransE/DistMult/ConvE)")

        return score

class CompGCNCov(nn.Module):

    """``CompGCN`` :cite:`CompGCN` 图神经网络模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: typing.Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        bias: bool = True,
        drop_rate: float = 0.,
        opn: str = 'corr'):

        """创建 CompGCN 对象。
        
        :param in_channels: 输入的特征维度
        :type in_channels: int
        :param out_channels: 输出的特征维度
        :type out_channels: int
        :param act: 激活函数
        :type act: typing.Callable[[torch.Tensor], torch.Tensor]
        :param bias: 是否有偏置
        :type bias: bool
        :param drop_rate: Dropout rate
        :type drop_rate: float
        :param opn: 组成运算符：'mult'、'sub'、'corr'
        :type opn: str
		"""

        super(CompGCNCov, self).__init__()

        #: 输入的特征维度
        self.in_channels: int = in_channels
        #: 输出的特征维度
        self.out_channels: int = out_channels
        self.rel_wt = None
        #: 关系嵌入向量
        self.rel: torch.nn.parameter.Parameter = None
        #: 组成运算符：'mult'、'sub'、'corr'
        self.opn: str = opn
        #：图神经网络的权重矩阵，用于原始关系
        self.in_w: torch.nn.parameter.Parameter = self.get_param([in_channels, out_channels])
        #：图神经网络的权重矩阵，用于相反关系
        self.out_w: torch.nn.parameter.Parameter = self.get_param([in_channels, out_channels])
        #: 用于原始关系和相反关系转换后输出结果的 Dropout
        self.drop: torch.nn.Dropout = nn.Dropout(drop_rate)
        #: 自循环关系嵌入向量的转换矩阵
        self.loop_rel: torch.nn.parameter.Parameter = self.get_param([1, in_channels])
        #：图神经网络的权重矩阵，用于自循环关系
        self.loop_w: torch.nn.parameter.Parameter = self.get_param([in_channels, out_channels])
        #: 偏置
        self.bias: torch.nn.Parameter = nn.Parameter(torch.zeros(out_channels)) if bias else None
        #: BatchNorm
        self.bn: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(out_channels)
        #: 激活函数
        self.act: typing.Callable[[torch.Tensor], torch.Tensor] = act
        #: 关系嵌入向量的转换矩阵
        self.w_rel: torch.nn.parameter.Parameter = self.get_param([in_channels, out_channels])

    def get_param(
        self,
        shape: list[int]) -> torch.nn.parameter.Parameter:

        """获得权重矩阵。
        
        :param shape: 权重矩阵的 shape
        :type shape: list[int]
        :returns: 权重矩阵
        :rtype: torch.nn.parameter.Parameter
        """

        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(
        self,
        graph: dgl.DGLGraph,
        ent_emb: torch.nn.parameter.Parameter,
        rel_emb: torch.nn.parameter.Parameter,
        edge_type: torch.Tensor,
        edge_norm: torch.Tensor) -> tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]:
		
	"""
 	定义每次调用时执行的计算。
  	:py:class:`torch.nn.Module` 子类必须重写 :py:meth:`torch.nn.Module.forward`。
   
   	:param graph: 子图
    	:type graph: dgl.DGLGraph
     	:param ent_emb: 实体嵌入向量
      	:type ent_emb: torch.nn.parameter.Parameter
       	:param rel_emb: 关系嵌入向量
	:type rel_emb: torch.nn.parameter.Parameter
 	:param edge_type: 关系 ID
  	:type edge_type: torch.Tensor
   	:param norm: 关系的归一化系数
    	:type norm: torch.Tensor
     	:returns: 更新后的实体嵌入和关系嵌入
      	:rtype: tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]
       	"""

        graph = graph.local_var()
        graph.ndata['h'] = ent_emb
        graph.edata['type'] = edge_type
        graph.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_emb
        else:
            self.rel = torch.mm(self.rel_wt, rel_emb)
        graph.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        ent_emb = graph.ndata.pop('h') + torch.mm(self.comp(ent_emb, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            ent_emb = ent_emb + self.bias
        ent_emb = self.bn(ent_emb)

        return self.act(ent_emb), torch.matmul(self.rel, self.w_rel)

    def message_func(self, edges: dgl.udf.EdgeBatch):

        """
        消息函数。
        """

        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)
        return {'msg': msg}

    def comp(
        self,
        h: torch.Tensor,
        r: torch.Tensor) -> torch.Tensor:

        """组成运算：'mult'、'sub'、'corr'
        
        :param h: 头实体嵌入向量
        :type h: torch.Tensor
        :param r: 关系嵌入向量
        :type r: torch.Tensor
        :returns: 组合后的边数据
        :rtype: torch.Tensor
        """

        def com_mult(a, b):

            """复数乘法"""

            r1, i1 = a.real, a.imag
            r2, i2 = b.real, b.imag
            real = r1 * r2 - i1 * i2
            imag = r1 * i2 + i1 * r2
            return torch.complex(real, imag)

        def conj(a):

            """共轭复数"""

            a.imag = -a.imag
            return a

        def ccorr(a, b):

            """corr 运算"""

            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), a.shape[-1])

        if self.opn == 'mult':
            return h * r
        elif self.opn == 'sub':
            return h - r
        elif self.opn == 'corr':
            return ccorr(h, r.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def reduce_func(self, nodes: dgl.udf.NodeBatch):

        """聚合函数"""
        
        return {'h': self.drop(nodes.data['h']) / 3}

def get_compgcn_hpo_config() -> dict[str, dict[str, typing.Any]]:
    
    """返回 :py:class:`CompGCN` 的默认超参数优化配置。
    
    默认配置为::
	
        parameters_dict = {
            'model': {
                'value': 'CompGCN'
            },
            'dim': {
                'values': [100, 150, 200]
            },
            'opn': {
                'value': 'mult'
            },
            'fet_drop': {
                'value': 0.2
            },
            'hid_drop': {
                'value': 0.3
            },
            'margin': {
                'value': 40.0
            },
            'decoder_model': {
                'value': 'ConvE'
            }
        }
        
    :returns: :py:class:`CompGCN` 的默认超参数优化配置
    :rtype: dict[str, dict[str, typing.Any]]
    """
    
    parameters_dict = {
        'model': {
            'value': 'CompGCN'
        },
        'dim': {
            'values': [100, 150, 200]
        },
        'opn': {
            'value': 'mult'
        },
        'fet_drop': {
            'value': 0.2
        },
        'hid_drop': {
            'value': 0.3
        },
        'margin': {
            'value': 40.0
        },
        'decoder_model': {
            'value': 'ConvE'
        }
    }
    
    return parameters_dict
