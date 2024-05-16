"""
**RESCAL-FB15K237-single-gpu** ||
`RESCAL-FB15K237-single-gpu-wandb <single_gpu_rescal_FB15K237_wandb.html>`_ ||
`RESCAL-FB15K237-single-gpu-hpo <single_gpu_rescal_FB15K237_hpo.html>`_

RESCAL-FB15K237-single-gpu
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 16, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 16, 2024

这一部分介绍如何用一个 GPU 在 ``FB15K237`` 知识图谱上训练 ``RESCAL`` :cite:`RESCAL`。

导入数据
-----------------
pybind11-OpenKE 有 1 个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader
from pybind11_ke.module.model import RESCAL
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = "../../benchmarks/FB15K237/", 
	batch_size = 2048,
	neg_ent = 25,
	test = True,
	test_batch_size = 10,
	num_workers = 16
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.RESCAL`，它是很多张量分解模型改进的基础。

# define the model
rescal = RESCAL(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = 50
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 TransE 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.MarginLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = rescal, 
	loss = MarginLoss(margin = 1.0)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.Trainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Trainer.run` 函数进行模型学习。

# test the model
tester = Tester(model = rescal, data_loader = dataloader, use_tqdm = False,
                use_gpu = True, device = 'cuda:1')

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(), epochs = 1000,
	lr = 0.1, opt_method = 'adagrad', use_gpu = True, device = 'cuda:1',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100,
	save_path = '../../checkpoint/rescal.pth', use_wandb = False)
trainer.run()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/RESCAL/single_gpu_rescal_FB15K237.txt>`_ 下载。

######################################################################
# --------------
#