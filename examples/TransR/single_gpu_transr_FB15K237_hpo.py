"""
`TransR-FB15K237-single-gpu <single_gpu_transr_FB15K237.html>`_ ||
`TransR-FB15K237-single-gpu-wandb <single_gpu_transr_FB15K237_wandb.html>`_ ||
**TransR-FB15K237-single-gpu-hpo** ||
`TransR-FB15K237-multigpu <multigpu_transr_FB15K237.html>`_

TransR-FB15K237-single-gpu-hpo
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 13, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 13, 2024

这一部分介绍如何用一个 GPU 在 ``FB15K237`` 知识图谱上寻找 ``TransR`` :cite:`TransR` 的超参数。

定义训练数据加载器超参数优化范围
---------------------------------------------------------
"""

import pprint
from pybind11_ke.data import get_kge_data_loader_hpo_config
from pybind11_ke.module.model import get_transr_hpo_config
from pybind11_ke.module.loss import get_margin_loss_hpo_config
from pybind11_ke.module.strategy import get_negative_sampling_hpo_config
from pybind11_ke.config import get_tester_hpo_config
from pybind11_ke.config import get_trainer_hpo_config
from pybind11_ke.config import set_hpo_config, start_hpo_train

######################################################################
# :py:func:`pybind11_ke.data.get_kge_data_loader_hpo_config` 将返回
# :py:class:`pybind11_ke.data.KGEDataLoader` 的默认超参数优化范围，
# 你可以修改数据目录等信息。

data_loader_config = get_kge_data_loader_hpo_config()
print("data_loader_config:")
pprint.pprint(data_loader_config)
print()

data_loader_config.update({
    'in_path': {
        'value': '../../benchmarks/FB15K237/'
    },
    'test_batch_size': {
        'value': 5
    },
})

######################################################################
# --------------
#

################################
# 定义模型超参数优化范围
# ---------------------------------------------------------
# :py:func:`pybind11_ke.module.model.get_transr_hpo_config` 返回了
# :py:class:`pybind11_ke.module.model.TransR` 的默认超参数优化范围。

# set the hpo config
kge_config = get_transr_hpo_config()
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
    sweep_name = "TransR_FB15K237",
    data_loader_config = data_loader_config,
    kge_config = kge_config,
    loss_config = loss_config,
    strategy_config = strategy_config,
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
# .. Note:: 上述代码的运行报告可以从 `此处 </zh-cn/latest/_static/pdfs/examples/TransR/TransR超参数搜索示例（一）.pdf>`_ 下载。

######################################################################
# --------------
#