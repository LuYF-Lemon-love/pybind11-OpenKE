# coding:utf-8
#
# pybind11_ke/module/loss/Cross_Entropy_Loss.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 19, 2024
#
# 该脚本定义了 Cross_Entropy_Loss 类.

import torch
import torch.nn as nn

class Cross_Entropy_Loss(nn.Module):

    """Binary CrossEntropyLoss 

    Attributes:
        args: Some pre-set parameters, etc 
        model: The KG model for training.
    """

    def __init__(self, model):
        super(Cross_Entropy_Loss, self).__init__()
        self.model = model
        self.loss = torch.nn.BCELoss()

    def forward(self, pred, label):

        """Creates a criterion that measures the Binary Cross Entropy between the target and
        the input probabilities. In math:

        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],
        
        Args:
            pred: The score of all samples.
            label: Vectors used to distinguish positive and negative samples.
        Returns:
            loss: The training loss for back propagation.
        """
        
        loss = self.loss(pred, label)
        return loss