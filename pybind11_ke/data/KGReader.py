# coding:utf-8
#
# pybind11_ke/data/KGReader.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 17, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2024
#
# 从文件中读取知识图谱.

"""
KGReader - 从文件中读取知识图谱。
"""

import os
import numpy as np
from collections import defaultdict as ddict

class KGReader:

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

        """创建 KGReader 对象。

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
        """

        #: 数据集目录
        self.in_path: str = in_path
        #: entity2id.txt
        self.ent_file: str = ent_file
        #: relation2id.txt
        self.rel_file: str = rel_file
        #: train2id.txt
        self.train_file: str = train_file
        #: valid2id.txt
        self.valid_file: str = valid_file
        #: test2id.txt
        self.test_file: str = test_file

        #: 实体的个数
        self.ent_tol: int = 0
        #: 关系的个数
        self.rel_tol: int = 0
        #: 训练集三元组的个数
        self.train_tol: int = 0
        #: 验证集三元组的个数
        self.valid_tol: int = 0
        #: 测试集三元组的个数
        self.test_tol: int = 0

        #: 实体->ID
        self.ent2id: dict = {}
        #: 关系->ID
        self.rel2id: dict = {}
        #: ID->实体
        self.id2ent: dict = {}
        #: ID->关系
        self.id2rel: dict = {}

        #: 训练集三元组
        self.train_triples: list[tuple[int, int, int]] = []
        #: 验证集三元组
        self.valid_triples: list[tuple[int, int, int]] = []
        #: 测试集三元组
        self.test_triples: list[tuple[int, int, int]] = []
        #: 知识图谱所有三元组
        self.all_true_triples: set[tuple[int, int, int]] = set()

        #: 训练集中所有 h-r 对对应的 t 集合
        self.hr2t_train: ddict[set] = ddict(set)
        #: 训练集中所有 r-t 对对应的 h 集合
        self.rt2h_train: ddict[set] = ddict(set)

        self.get_id()
        self.get_triples_id()
    
    def get_id(self):

        """读取 :py:attr:`ent_file` 文件和 :py:attr:`rel_file` 文件。"""
        
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

        """读取 :py:attr:`train_file` 文件、 :py:attr:`valid_file` 文件和 :py:attr:`test_file` 文件。"""
        
        with open(os.path.join(self.in_path, self.train_file)) as f:
            self.train_tol = (int)(f.readline())
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

        """获得 :py:attr:`hr2t_train` 和 :py:attr:`rt2h_train` 。"""
        
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))
            
    def get_hr_train(self):

        """用于 ``CompGCN`` :cite:`CompGCN` 训练，因为 ``CompGCN`` :cite:`CompGCN` 的组合运算仅需要头实体和关系。
        
        如果想获得更详细的信息请访问 :ref:`CompGCN <compgcn>`。
        """
        
        self.t_triples = self.train_triples
        self.train_triples = [(hr, list(t)) for (hr,t) in self.hr2t_train.items()]
        
    def get_train(self) -> list[tuple[int, int, int]]:

        """
        返回训练集三元组。

        :returns: :py:attr:`train_triples`
        :rtype: list[tuple[int, int, int]]
        """

        return self.train_triples

    def get_valid(self) -> list[tuple[int, int, int]]:

        """
        返回验证集三元组。

        :returns: :py:attr:`valid_triples`
        :rtype: list[tuple[int, int, int]]
        """

        return self.valid_triples

    def get_test(self) -> list[tuple[int, int, int]]:

        """
        返回测试集三元组。

        :returns: :py:attr:`test_triples`
        :rtype: list[tuple[int, int, int]]
        """

        return self.test_triples
    
    def get_all_true_triples(self) -> set[tuple[int, int, int]]:

        """
        返回知识图谱所有三元组。

        :returns: :py:attr:`all_true_triples`
        :rtype: set[tuple[int, int, int]]
        """

        return self.all_true_triples 