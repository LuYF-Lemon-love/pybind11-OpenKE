"""
**RGCN-FB15K237-single-gpu** ||
`RGCN-FB15K237-single-gpu-wandb <single_gpu_rgcn_FB15K237_wandb.html>`_ ||
`RGCN-FB15K237-single-gpu-hpo <single_gpu_rgcn_FB15K237_hpo.html>`_

RGCN-FB15K237-single-gpu
=====================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 20, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 20, 2024

这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``R-GCN`` :cite:`R-GCN`。

导入数据
-----------------
pybind11-OpenKE 有一个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader, RGCNSampler, RGCNTestSampler
from pybind11_ke.module.model import RGCN
from pybind11_ke.module.loss import RGCNLoss
from pybind11_ke.module.strategy import RGCNSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = KGEDataLoader(
	in_path = "../../benchmarks/FB15K237/",
	batch_size = 60000,
	neg_ent = 10,
	test = True,
	test_batch_size = 100,
	num_workers = 16,
    train_sampler = RGCNSampler,
    test_sampler = RGCNTestSampler
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.RGCN`，它提出于 2017 年，是第一个图神经网络模型，

# define the model
rgcn = RGCN(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = 500,
	num_layers = 2
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``R-GCN`` :cite:`R-GCN` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.RGCNLoss`，
# :py:class:`pybind11_ke.module.strategy.RGCNSampling` 对
# :py:class:`pybind11_ke.module.loss.RGCNLoss` 进行了封装。

# define the loss function
model = RGCNSampling(
	model = rgcn,
	loss = RGCNLoss(model = rgcn, regularization = 1e-5)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.Trainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Trainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`pybind11_ke.config.Tester`，
# 使得训练器能够在训练过程中评估模型。

# test the model
tester = Tester(model = rgcn, data_loader = dataloader, use_tqdm = False,
                use_gpu = True, device = 'cuda:0')

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 10000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 500, log_interval = 500,
	save_interval = 500, save_path = '../../checkpoint/rgcn.pth'
)
trainer.run()