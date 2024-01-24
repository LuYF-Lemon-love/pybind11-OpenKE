"""
`SimplE-WN18RR-single-gpu-wandb <single_gpu_simple_WN18RR_wandb.html>`_ ||
**SimplE-WN18RR-single-gpu-hpo**

SimplE-WN18RR-single-gpu-hpo
====================================================================

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上寻找 ``SimplE`` :cite:`SimplE` 的超参数。

定义训练数据加载器超参数优化范围
---------------------------------------------------------
"""

import pprint
from pybind11_ke.data import get_train_data_loader_hpo_config
from pybind11_ke.module.model import get_simple_hpo_config
from pybind11_ke.module.loss import get_softplus_loss_hpo_config
from pybind11_ke.module.strategy import get_negative_sampling_hpo_config
from pybind11_ke.data import get_test_data_loader_hpo_config
from pybind11_ke.config import get_tester_hpo_config
from pybind11_ke.config import get_trainer_hpo_config
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
        'value': '../../benchmarks/WN18RR/'
    },
    'threads': {
        'value': 2
    },
})

######################################################################
# --------------
#

################################
# 定义模型超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.model.get_simple_hpo_config` 返回了
# :py:class:`pybind11_ke.module.model.SimplE` 的默认超参数优化范围。

# set the hpo config
kge_config = get_simple_hpo_config()
print("kge_config:")
pprint.pprint(kge_config)
print()

######################################################################
# --------------
#

################################
# 定义损失函数超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.loss.get_softplus_loss_hpo_config` 返回了
# :py:class:`pybind11_ke.module.loss.SoftplusLoss` 的默认超参数优化范围。

# set the hpo config
loss_config = get_softplus_loss_hpo_config()
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
# 定义评估器超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.config.get_tester_hpo_config` 返回了
# :py:class:`pybind11_ke.config.Tester` 的默认超参数优化范围。

# set the hpo config
tester_config = get_tester_hpo_config()
print("tester_config:")
pprint.pprint(tester_config)
print()

######################################################################
# --------------
#

################################
# 定义训练器超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.config.get_trainer_hpo_config` 返回了
# :py:class:`pybind11_ke.config.Trainer` 的默认超参数优化范围。

# set the hpo config
trainer_config = get_trainer_hpo_config()
print("trainer_config:")
pprint.pprint(trainer_config)
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
    sweep_name = "SimplE_WN18RR",
    train_data_loader_config = train_data_loader_config,
    kge_config = kge_config,
    loss_config = loss_config,
    strategy_config = strategy_config,
    test_data_loader_config = test_data_loader_config,
    tester_config = tester_config,
    trainer_config = trainer_config)
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
start_hpo_train(config=sweep_config, count=3)

######################################################################
# .. figure:: /_static/images/examples/SimplE/SimplE-WN18RR-HPO.png
#      :align: center
#      :height: 300
#
#      超参数优化结果

######################################################################
# --------------
#