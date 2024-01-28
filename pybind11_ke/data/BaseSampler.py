# coding:utf-8
#
# pybind11_ke/data/RevSampler.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 28, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2024
#
# 为 KGReader 增加相反关系，用于图神经网络模型.

"""
RevSampler - 为 KGReader 增加相反关系，用于图神经网络模型。
"""

import torch
import numpy as np
from .KGReader import KGReader

class BaseSampler(KGReader):
    
    """平移模型和语义匹配模型的采样器的基类。
    """
    
    def __init__(
        self,
        in_path: str = "./",
        ent_file: str = "entity2id.txt",
        rel_file: str = "relation2id.txt",
        train_file: str = "train2id.txt"):

        """创建 RevSampler 对象。

        :param in_path: 数据集目录
        :type in_path: str
        :param ent_file: entity2id.txt
        :type ent_file: str
        :param rel_file: relation2id.txt
        :type rel_file: str
        :param train_file: train2id.txt
        :type train_file: str
        """
        
        super().__init__(
            in_path=in_path,
            ent_file=ent_file,
            rel_file=rel_file,
            train_file=train_file
        )

        self.get_hr2t_rt2h_from_train()

    def head_batch(self, h, r, t, neg_size=None):
        
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def corrupt_head(self, t, r, num_max=1):
        
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        
        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def tail_batch(self, h, r, t, neg_size=None):
        
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]
        
    def corrupt_tail(self, h, r, num_max=1):
        
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        
        tmp = torch.randint(low=0, high=self.ent_tol, size=(num_max,)).numpy()
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg