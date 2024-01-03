"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
**TransE-FB15K-single-gpu-hpo**
`TransE-FB15K-multigpu <multigpu_transe_FB15K.html>`_ ||
`TransE-FB15K-multigpu-wandb <multigpu_transe_FB15K_wandb.html>`_

TransE-FB15K-single-gpu-hpo
====================================================================

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上寻找 ``TransE`` :cite:`TransE` 的超参数。

导入训练数据加载器超参数默认配置
---------------------------------------------------------
pybind11-OpenKE 有 1 个工具用于导入超参数默认配置: :py:func:`pybind11_ke.config.get_hpo_config`。
"""

import pprint
from pybind11_ke.data import get_train_data_loader_hpo_config
from pybind11_ke.config import set_hpo_config, start_hpo_train

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。
# :py:func:`pybind11_ke.data.get_train_data_loader_hpo_config` 将返回训练数据加载的默认优化参数，
# 你可以修改数据目录等信息。

train_data_loader_config = get_train_data_loader_hpo_config()
print("train_data_loader_config:")
pprint.pprint(train_data_loader_config)

train_data_loader_config.update({
    'in_path': {
        'value': './benchmarks/FB15K/'
    }
})

######################################################################
# --------------
#

################################
# 设置超参数优化参数
# ---------------------------------------------------------
# :py:func:`pybind11_ke.config.set_hpo_config` 可以设置超参数优化参数。

# set the hpo config
sweep_config = set_hpo_config(train_data_loader_config=train_data_loader_config)
print("sweep_config:")
pprint.pprint(sweep_config)

######################################################################
# --------------
#

################################
# 开始超参数优化
# ---------------------------------------------------------
# :py:func:`pybind11_ke.config.start_hpo_train` 可以开始超参数优化。

# start hpo
start_hpo_train(config=sweep_config)

######################################################################
# --------------
#