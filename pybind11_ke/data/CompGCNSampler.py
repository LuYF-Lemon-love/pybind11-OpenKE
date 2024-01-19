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

import torch
import numpy as np
from .GraphSampler import GraphSampler

class CompGCNSampler(GraphSampler):
    
    """Graph based sampling in neural network.

    Attributes:
        relation: The relation of sampled triples.
        triples: The sampled triples.
        graph: The graph structured sampled triples by dgl.graph in DGL.
        norm: The edge norm in graph.
        label: Mask the false tail as negative samples.
    """

    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt",
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
        :param valid_file: valid2id.txt
        :type valid_file: str
        :param test_file: test2id.txt
        :type test_file: str
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
            valid_file=valid_file,
            test_file=test_file,
            batch_size=batch_size,
            neg_ent=neg_ent
        )

        #: batch size
        self.batch_size: int = batch_size

        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None
        
        super().get_hr_train()
        
        self.graph, self.relation, self.norm = \
            self.build_graph(self.ent_tol, np.array(self.t_triples).transpose(), -0.5)

    def sampling(self, pos_hr_t):
        
        """Graph based n_hop neighbours in neural network.

        Args:
            pos_hr_t: The triples(hr, t) used to be sampled.

        Returns:
            batch_data: The training data.
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

    def node_norm_to_edge_norm(self, graph, node_norm):

        """Calculating the normalization edge weight.

        Args:
            graph: The graph structured sampled triples by dgl.graph in DGL.
            node_norm: The node weight of normalization.

        Returns:
            norm: The edge weight of normalization.
        """
        
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        norm = graph.edata.pop('norm').squeeze()
        return norm