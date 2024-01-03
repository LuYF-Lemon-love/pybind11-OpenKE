"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
**TransE-FB15K-single-gpu-hpo**
`TransE-FB15K-multigpu <multigpu_transe_FB15K.html>`_ ||
`TransE-FB15K-multigpu-wandb <multigpu_transe_FB15K_wandb.html>`_

TransE-FB15K-single-gpu-hpo
====================================================================

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上寻找 ``TransE`` :cite:`TransE` 的超参数。

导入超参数默认配置
------------------------
pybind11-OpenKE 有 1 个工具用于导入超参数默认配置: :py:func:`pybind11_ke.config.get_hpo_config`。
"""

from pybind11_ke.config import get_hpo_config

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
sweep_config = get_hpo_config()

######################################################################
# --------------
#