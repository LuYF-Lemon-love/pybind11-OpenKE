# coding:utf-8
#
# pybind11_ke/module/model/RGCN.py
# 
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# 
# 该头文件定义了 RGCN.

"""
RGCN - 第一个平移模型，简单而且高效。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

class RGCN(nn.Module):

    """RGCN
    """

    def __init__(
        self,
        ent_tot: int,
        rel_tot: int,
        dim: int,
        num_layers: int):

        super(RGCN, self).__init__()

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.num_layers = num_layers

        self.ent_emb = None
        self.rel_emb = None 
        self.RGCN = None 
        self.Loss_emb = None
        self.build_model()

    def build_model(self):

        self.ent_emb = nn.Embedding(self.ent_tot, self.dim)

        self.rel_emb = nn.Parameter(torch.Tensor(self.rel_tot, self.dim))

        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.RGCN = nn.ModuleList()
        for idx in range(self.num_layers):
            RGCN_idx = self.build_hidden_layer(idx)
            self.RGCN.append(RGCN_idx)

    def build_hidden_layer(self, idx):

        act = F.relu if idx < self.num_layers - 1 else None
        return RelGraphConv(self.dim, self.dim, self.rel_tot, "bdd",
                    num_bases=100, activation=act, self_loop=True, dropout=0.2)
        
    def forward(self, graph, ent, rel, norm, triples, mode='single'):
        
        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = self.distmult_score_func(head_emb, rela_emb, tail_emb, mode)

        return score

    def tri2emb(self, embedding, triples, mode="single"):

        rela_emb = self.rel_emb[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
        head_emb = embedding[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim] 
        tail_emb = embedding[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]

        if mode == "head-batch" or mode == "head_predict":
            head_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        elif mode == "tail-batch" or mode == "tail_predict":
            tail_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        return head_emb, rela_emb, tail_emb

    def distmult_score_func(self, head_emb, relation_emb, tail_emb, mode):

        if mode == 'head-batch':
            score = head_emb * (relation_emb * tail_emb)
        else:
            score = (head_emb * relation_emb) * tail_emb

        score = score.sum(dim = -1)
        return score

    def get_score(self, batch, mode):

        triples    = batch['positive_sample']
        graph      = batch['graph']
        ent        = batch['entity']
        rel        = batch['rela']
        norm       = batch['norm']

        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = self.distmult_score_func(head_emb, rela_emb, tail_emb, mode)

        return score