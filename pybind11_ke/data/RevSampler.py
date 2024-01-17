# coding:utf-8
#
# pybind11_ke/data/RevSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 15, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
#
# 增加相反关系.

import os
import torch
import numpy as np
from .Sampler import Sampler

class RevSampler(Sampler):
    
    """增加相反关系
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt"):
        
        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file
        )
        self.add_reverse_relation()
        self.add_reverse_triples()
        self.get_hr2t_rt2h_from_train()
        
    def add_reverse_relation(self):
        
        with open(os.path.join(self.in_path, self.rel_file)) as f:
            tol = (int)(f.readline())
            for line in f:
                relation, rid = line.strip().split("\t")
                self.rel2id[relation + "_reverse"] = int(rid) + tol
                self.id2rel[int(rid) + tol] = relation + "_reverse"
        self.rel_tol = len(self.rel2id)
        
    def add_reverse_triples(self):

        tol = int(self.rel_tol / 2)
        
        with open(os.path.join(self.in_path, self.train_file)) as f:
            f.readline()
            for line in f:
                h, t, r = line.strip().split()
                self.train_triples.append(
                    (int(t), int(r) + tol, int(h))
                )
                
        with open(os.path.join(self.in_path, self.valid_file)) as f:
            f.readline()
            for line in f:
                h, t, r = line.strip().split()
                self.valid_triples.append(
                    (int(t), int(r) + tol, int(h))
                )
                
        with open(os.path.join(self.in_path, self.test_file)) as f:
            f.readline()
            for line in f:
                h, t, r = line.strip().split()
                self.test_triples.append(
                    (int(t), int(r) + tol, int(h))
                )
                
        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )
    
    def corrupt_head(self, t, r, num_max=1):
        
        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):

        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):

        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]