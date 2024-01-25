"""
`RGCN-FB15K237-single-gpu <single_gpu_rgcn_FB15K237.html>`_ ||
**RGCN-FB15K237-single-gpu-wandb** ||
`RGCN-FB15K237-single-gpu-hpo <single_gpu_rgcn_FB15K237_hpo.html>`_

RGCN-FB15K237-single-gpu-wandb
=====================================================

这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``R-GCN`` :cite:`R-GCN`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有一个工具用于导入数据: :py:class:`pybind11_ke.data.GraphDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import GraphDataLoader
from pybind11_ke.module.model import RGCN
from pybind11_ke.module.loss import RGCNLoss
from pybind11_ke.module.strategy import RGCNSampling
from pybind11_ke.config import GraphTrainer, GraphTester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="rgcn",
	config=dict(
		in_path = '../../benchmarks/FB15K237/',
		batch_size = 60000,
		neg_ent = 10,
		test = True,
		test_batch_size = 100,
		num_workers = 16,
		dim = 500,
		num_layers = 2,
		regularization = 1e-5,
		use_gpu = True,
		device = 'cuda:0',
		epochs = 10000,
		lr = 0.0001,
		valid_interval = 500,
		log_interval = 500,
		save_interval = 500,
		save_path = '../../checkpoint/rgcn.pth'
	)
)

config = wandb_logger.config

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.GraphDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = GraphDataLoader(
	in_path = config.in_path,
	batch_size = config.batch_size,
	neg_ent = config.neg_ent,
	test = config.test,
	test_batch_size = config.test_batch_size,
	num_workers = config.num_workers
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
	ent_tol = dataloader.train_sampler.ent_tol,
	rel_tol = dataloader.train_sampler.rel_tol,
	dim = config.dim,
	num_layers = config.num_layers
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
	loss = RGCNLoss(model = rgcn, regularization = config.regularization)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.GraphTrainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.GraphTrainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`pybind11_ke.config.GraphTester`，
# 使得训练器能够在训练过程中评估模型；:py:class:`pybind11_ke.config.GraphTester` 使用
# :py:class:`pybind11_ke.data.GraphDataLoader` 作为数据采样器。

# test the model
tester = GraphTester(model = rgcn, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = GraphTrainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs, lr = config.lr, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval, log_interval = config.log_interval,
	save_interval = config.save_interval, save_path = config.save_path, use_wandb = True
)
trainer.run()

######################################################################
# .. figure:: /_static/images/examples/RGCN/RGCN-FB15K237-Loss.png
#      :align: center
#      :height: 300
#
#      训练过程中损失值的变化

######################################################################
# .. figure:: /_static/images/examples/RGCN/RGCN-FB15K237-MR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MR 的变化

######################################################################
# .. figure:: /_static/images/examples/RGCN/RGCN-FB15K237-MRR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MRR 的变化

######################################################################
# .. figure:: /_static/images/examples/RGCN/RGCN-FB15K237-Hit.png
#      :align: center
#      :height: 300
#
#      训练过程中 Hits@3、Hits@3 和 Hits@10 的变化

######################################################################
# --------------
#