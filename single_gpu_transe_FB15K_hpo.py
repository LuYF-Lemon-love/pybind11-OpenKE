"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
**TransE-FB15K-single-gpu-hpo**
`TransE-FB15K-multigpu <multigpu_transe_FB15K.html>`_ ||
`TransE-FB15K-multigpu-wandb <multigpu_transe_FB15K_wandb.html>`_

TransE-FB15K-single-gpu-hpo
====================================================================

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上寻找 ``TransE`` :cite:`TransE` 的超参数。

定义训练数据加载器超参数优化范围
---------------------------------------------------------
pybind11-OpenKE 有 1 个工具用于导入超参数默认配置: :py:func:`pybind11_ke.config.get_hpo_config`。
"""

import pprint
from pybind11_ke.data import get_train_data_loader_hpo_config
from pybind11_ke.module.model import get_transe_hpo_config
from pybind11_ke.module.loss import get_margin_loss_hpo_config
from pybind11_ke.module.strategy import get_negative_sampling_hpo_config
from pybind11_ke.data import get_test_data_loader_hpo_config
from pybind11_ke.config import set_hpo_config, start_hpo_train

######################################################################
# :py:func:`pybind11_ke.data.get_train_data_loader_hpo_config` 将返回
# :py:class:`pybind11_ke.data.TrainDataLoader` 的默认超参数优化范围，
# 你可以修改数据目录等信息。

train_data_loader_config = get_train_data_loader_hpo_config()
print("train_data_loader_config:")
pprint.pprint(train_data_loader_config)
print()

train_data_loader_config.update({
    'in_path': {
        'value': './benchmarks/FB15K/'
    }
})

######################################################################
# --------------
#

################################
# 定义模型超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.model.get_transe_hpo_config` 返回了
# :py:class:`pybind11_ke.module.model.TransE` 的默认超参数优化范围。

# set the hpo config
kge_config = get_transe_hpo_config()
print("kge_config:")
pprint.pprint(kge_config)
print()

######################################################################
# --------------
#

################################
# 定义损失函数超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.loss.get_margin_loss_hpo_config` 返回了
# :py:class:`pybind11_ke.module.loss.MarginLoss` 的默认超参数优化范围。

# set the hpo config
loss_config = get_margin_loss_hpo_config()
print("loss_config:")
pprint.pprint(loss_config)
print()

######################################################################
# --------------
#

################################
# 定义训练策略超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.strategy.get_negative_sampling_hpo_config` 返回了
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 的默认超参数优化范围。

# set the hpo config
strategy_config = get_negative_sampling_hpo_config()
print("strategy_config:")
pprint.pprint(strategy_config)
print()

######################################################################
# --------------
#

################################
# 定义测试数据加载器超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.data.get_test_data_loader_hpo_config` 返回了
# :py:class:`pybind11_ke.data.TestDataLoader` 的默认超参数优化范围。

# set the hpo config
test_data_loader_config = get_test_data_loader_hpo_config()
print("test_data_loader_config:")
pprint.pprint(test_data_loader_config)
print()

######################################################################
# --------------
#

################################
# 设置超参数优化参数
# ---------------------------------------------------------
# :py:func:`pybind11_ke.config.set_hpo_config` 可以设置超参数优化参数。

# set the hpo config
sweep_config = set_hpo_config(
    train_data_loader_config = train_data_loader_config,
    kge_config = kge_config,
    loss_config = loss_config,
    strategy_config = strategy_config,
    test_data_loader_config = test_data_loader_config)
print("sweep_config:")
pprint.pprint(sweep_config)
print()

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