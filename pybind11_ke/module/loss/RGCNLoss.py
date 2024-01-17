# coding:utf-8
#
# pybind11_ke/module/loss/RGCN_Loss.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
#
# 该脚本定义了 RGCNLoss 类.

import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCNLoss(nn.Module):

    def __init__(self, model, regularization):
        super(RGCNLoss, self).__init__()
        self.model = model
        self.regularization = regularization
    
    def reg_loss(self): 
        return torch.mean(self.model.Loss_emb.pow(2)) + torch.mean(self.model.rel_emb.pow(2))

    def forward(self, score, labels):
         loss = F.binary_cross_entropy_with_logits(score, labels)
         regu = self.regularization * self.reg_loss()
         loss += regu
         return loss