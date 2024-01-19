"""
**RGCN-FB15K237-single-gpu**

RGCN-FB15K237-single-gpu
=====================================================
这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``R-GCN`` :cite:`R-GCN`。

导入数据
-----------------
pybind11-OpenKE 有一个工具用于导入数据: :py:class:`pybind11_ke.data.GraphDataLoader`。
"""

from pybind11_ke.data import CompGCNSampler, CompGCNTestSampler ,GraphDataLoader
from pybind11_ke.module.model import CompGCN
from pybind11_ke.module.loss import Cross_Entropy_Loss
from pybind11_ke.module.strategy import CompGCNSampling
from pybind11_ke.config import RGCNTrainer, RGCNTester

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.GraphDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = GraphDataLoader(
	in_path = "../../benchmarks/FB15K237/",
	batch_size = 2048,
	test_batch_size = 256,
	num_workers = 16,
	train_sampler = CompGCNSampler,
	test_sampler = CompGCNTestSampler
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
compgcn = CompGCN(
	ent_tot = dataloader.train_sampler.ent_tol,
	rel_tot = dataloader.train_sampler.rel_tol,
	dim = 100
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
model = CompGCNSampling(
	model = compgcn,
	loss = Cross_Entropy_Loss(model = compgcn),
	ent_tol = dataloader.train_sampler.ent_tol
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.RGCNTrainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.RGCNTrainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`pybind11_ke.config.RGCNTester`，
# 使得训练器能够在训练过程中评估模型；:py:class:`pybind11_ke.config.RGCNTester` 使用
# :py:class:`pybind11_ke.data.GraphDataLoader` 作为数据采样器。

# test the model
tester = RGCNTester(model = compgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")

# train the model
trainer = RGCNTrainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 2000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 50, log_interval = 50,
	# save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
)
trainer.run()