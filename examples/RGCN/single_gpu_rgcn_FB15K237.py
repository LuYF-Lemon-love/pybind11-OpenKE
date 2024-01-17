"""
**TransH-FB15K237-single-gpu** ||
`TransH-FB15K237-single-gpu-wandb <single_gpu_transh_FB15K237_wandb.html>`_ ||
`TransH-FB15K237-multigpu <multigpu_transh_FB15K237.html>`_ ||
`TransH-FB15K-multigpu-wandb <multigpu_transh_FB15K237_wandb.html>`_

RGCN-FB15K237-single-gpu
=====================================================
这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``TransH`` :cite:`TransH`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.data import GraphDataLoader
from pybind11_ke.module.model import RGCN
from pybind11_ke.module.loss import RGCNLoss
from pybind11_ke.module.strategy import RGCNSampling
from pybind11_ke.config import RGCNTrainer, RGCNTester

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = GraphDataLoader(
	in_path = "../../benchmarks/FB15K237/",
	batch_size = 60000,
	neg_ent = 10,
	test_batch_size = 300,
	num_workers = 16
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.TransH`，它提出于 2014 年，是第二个平移模型，
# 将关系建模为超平面上的平移操作。

# define the model
rgcn = RGCN(
	ent_tot = dataloader.train_sampler.ent_tol,
	rel_tot = dataloader.train_sampler.rel_tol,
	dim = 500,
	num_layers = 2
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``TransE`` :cite:`TransE` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.MarginLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

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
# 使得训练器能够在训练过程中评估模型；:py:class:`pybind11_ke.config.Tester` 使用
# :py:class:`pybind11_ke.data.TestDataLoader` 作为数据采样器。

# test the model
tester = RGCNTester(model = rgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0')

# train the model
trainer = RGCNTrainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 10000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 500, #log_interval = 10,
	# save_interval = 100, save_path = '../../checkpoint/transh.pth'
)
trainer.run()