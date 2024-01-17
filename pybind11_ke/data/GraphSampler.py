# coding:utf-8
#
# pybind11_ke/data/GraphSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
#
# RGCN 的数据采样器.

import dgl
import torch
import warnings
import numpy as np
from .RevSampler import RevSampler

warnings.filterwarnings("ignore")

class GraphSampler(RevSampler):

    """基本图神经网络采样器。
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

        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file
        )

        self.batch_size: int = batch_size
        self.neg_ent: int = neg_ent

        self.entity   = None
        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None

    def sampling(self, pos_triples):
        
        batch_data = {}
        
        pos_triples = np.array(pos_triples)
        pos_triples, self.entity = self.sampling_positive(pos_triples)
        head_triples = self.sampling_negative('head', pos_triples)
        tail_triples = self.sampling_negative('tail', pos_triples)
        self.triples = np.concatenate((pos_triples,head_triples,tail_triples))
        batch_data['entity']  = self.entity
        batch_data['triples'] = self.triples
        
        self.label = torch.zeros((len(self.triples),1))
        self.label[0 : self.batch_size] = 1
        batch_data['label'] = self.label
        
        split_size = int(self.batch_size * 0.5) 
        graph_split_ids = np.random.choice(
            self.batch_size,
            size=split_size, 
            replace=False
        )
        head,rela,tail = pos_triples.transpose()
        head = torch.tensor(head[graph_split_ids], dtype=torch.long).contiguous()
        rela = torch.tensor(rela[graph_split_ids], dtype=torch.long).contiguous()
        tail = torch.tensor(tail[graph_split_ids], dtype=torch.long).contiguous()
        self.graph, self.relation, self.norm = self.build_graph(len(self.entity), (head,rela,tail), -1)
        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    def sampling_positive(self, positive_triples):

        edges = np.random.choice(
            np.arange(len(positive_triples)),
            size = self.batch_size,
            replace=False
        )
        edges = positive_triples[edges]
        head, rela, tail = np.array(edges).transpose()
        entity, index = np.unique((head, tail), return_inverse=True) 
        head, tail = np.reshape(index, (2, -1))

        return np.stack((head,rela,tail)).transpose(), \
                torch.from_numpy(entity).view(-1,1).long()

    def sampling_negative(self, mode, pos_triples):

        neg_random = np.random.choice(
            len(self.entity), 
            size = self.neg_ent * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (self.neg_ent, 1))
        if mode == 'head':
            neg_samples[:,0] = neg_random
        elif mode == 'tail':
            neg_samples[:,2] = neg_random
        return neg_samples

    def build_graph(self, num_ent, triples, power):

        head, rela, tail = triples[0], triples[1], triples[2]
        graph = dgl.graph(([], []))
        graph.add_nodes(num_ent)
        graph.add_edges(head, tail)
        node_norm = self.comp_deg_norm(graph, power)
        edge_norm = self.node_norm_to_edge_norm(graph,node_norm)
        rela = torch.tensor(rela)
        return graph, rela, edge_norm

    def comp_deg_norm(self, graph, power=-1):

        graph = graph.local_var()
        in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        norm = in_deg.__pow__(power)
        norm[np.isinf(norm)] = 0
        return torch.from_numpy(norm)

    def node_norm_to_edge_norm(self, graph, node_norm):
        
        graph = graph.local_var()
        # convert to edge norm
        graph.ndata['norm'] = node_norm.view(-1,1)
        graph.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        return graph.edata['norm']