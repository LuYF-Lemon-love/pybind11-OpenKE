# coding:utf-8
#
# pybind11_ke/data/CompGCNSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2023
#
# 该脚本定义了 CompGCNSampler 类.

"""
CompGCNSampler - CompGCN 的数据采样器。
"""

import dgl
import torch
import typing
import numpy as np
from .GraphSampler import GraphSampler
from typing_extensions import override

class CompGCNSampler(GraphSampler):
    
    """``CompGCN`` :cite:`CompGCN` 的训练数据采样器。

    例子::

        from pybind11_ke.data import CompGCNSampler
        from torch.utils.data import DataLoader

        #: 训练数据采样器
        train_sampler: typing.Type[CompGCNSampler] = CompGCNSampler(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            batch_size=batch_size,
            neg_ent=neg_ent
        )

        train_dataloader = DataLoader(
            data_train,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_sampler.sampling,
        )
    """

    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        batch_size: int | None = None,
        neg_ent: int = 1):

        """创建 CompGCNSampler 对象。

        :param in_path: 数据集目录
        :type in_path: str
        :param ent_file: entity2id.txt
        :type ent_file: str
        :param rel_file: relation2id.txt
        :type rel_file: str
        :param train_file: train2id.txt
        :type train_file: str
        :param batch_size: batch size
        :type batch_size: int | None
        :param neg_ent: 对于 CompGCN 不起作用。
        :type neg_ent: int
        """

        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            batch_size=batch_size,
            neg_ent=neg_ent
        )
        
        super().get_hr_train()
        
        self.graph, self.relation, self.norm = \
            self.build_graph(self.ent_tol, np.array(self.t_triples).transpose(), -0.5)

    @override
    def sampling(
        self,
        pos_hr_t: list[tuple[tuple[int, int], list[int]]]) -> dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]:
        
        """``CompGCN`` :cite:`CompGCN` 的采样函数。

        :param pos_triples: 知识图谱中的正确三元组
        :type pos_triples: list[tuple[tuple[int, int], list[int]]]
        :returns: ``CompGCN`` :cite:`CompGCN` 的训练数据
        :rtype: dict[str, typing.Union[dgl.DGLGraph, torch.Tensor]]
        """

        batch_data = {}
        
        self.label = torch.zeros(self.batch_size, self.ent_tol)
        self.triples  = torch.LongTensor([hr for hr , _ in pos_hr_t])
        for id, hr_sample in enumerate([t for _ ,t in pos_hr_t]):
            self.label[id][hr_sample] = 1

        batch_data['sample']   = self.triples
        batch_data['label']    = self.label
        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    @override
    def node_norm_to_edge_norm(
        self,
        graph: dgl.DGLGraph,
        node_norm: torch.Tensor) -> torch.Tensor:

        """根据源节点和目标节点的度计算每条边的归一化系数。

        :param graph: 子图的节点数
        :type graph: dgl.DGLGraph
        :param node_norm: 节点的归一化系数
        :type node_norm: torch.Tensor
        :returns: 边的归一化系数
        :rtype: torch.Tensor
        """
        
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        norm = graph.edata.pop('norm').squeeze()
        return norm