# coding:utf-8
#
# pybind11_ke/data/Sampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 15, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 15, 2024
#
# 读取数据.

"""
Sampler - 读取数据集类。
"""

import os
import numpy as np
from collections import defaultdict as ddict

class Sampler:

    """从文件中读取知识图谱。
	"""
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt",
        valid_file: str = "valid2id.txt",
        test_file: str = "test2id.txt"):

        self.in_path: str = in_path
        self.ent_file: str = ent_file
        self.rel_file: str = rel_file
        self.train_file: str = train_file
        self.valid_file: str = valid_file
        self.test_file: str = test_file

        self.ent_tol: int = 0
        self.rel_tol: int = 0
        self.train_tot: int = 0
        self.valid_tol: int = 0
        self.test_tol: int = 0
        
        self.ent2id = {}
        self.rel2id = {}
        
        self.id2ent = {}
        self.id2rel = {}
        
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()
        
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.get_id()
        self.get_triples_id()
    
    def get_id(self):
        
        with open(os.path.join(self.in_path, self.ent_file)) as f:
            self.ent_tol = (int)(f.readline())
            for line in f:
                entity, eid = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity
        
        with open(os.path.join(self.in_path, self.rel_file)) as f:
            self.rel_tol = (int)(f.readline())
            for line in f:
                relation, rid = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation
    
    def get_triples_id(self):
        
        with open(os.path.join(self.in_path, self.train_file)) as f:
            self.train_tot = (int)(f.readline())
            for line in f:
                h, t, r = line.strip().split()
                self.train_triples.append((int(h), int(r), int(t)))
                
        with open(os.path.join(self.in_path, self.valid_file)) as f:
            self.valid_tol = (int)(f.readline())
            for line in f:
                h, t, r = line.strip().split()
                self.valid_triples.append((int(h), int(r), int(t)))
                
        with open(os.path.join(self.in_path, self.test_file)) as f:
            self.test_tol = (int)(f.readline())
            for line in f:
                h, t, r = line.strip().split()
                self.test_triples.append((int(h), int(r), int(t)))
                
        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )
        
    def get_hr2t_rt2h_from_train(self):
        
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))
            
    def get_hr_trian(self):
        
        self.t_triples = self.train_triples
        self.train_triples = [(hr, list(t)) for (hr,t) in self.hr2t_train.items()]
        
    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples
    
    def get_all_true_triples(self):
        return self.all_true_triples 