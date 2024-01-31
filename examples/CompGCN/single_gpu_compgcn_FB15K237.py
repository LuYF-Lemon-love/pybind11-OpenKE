"""
**CompGCN-FB15K237-single-gpu** ||
`CompGCN-FB15K237-single-gpu-wandb <single_gpu_compgcn_FB15K237_wandb.html>`_ ||
`CompGCN-FB15K237-single-gpu-hpo <single_gpu_compgcn_FB15K237_hpo.html>`_

CompGCN-FB15K237-single-gpu
=====================================================
这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``CompGCN`` :cite:`CompGCN`。

导入数据
-----------------
pybind11-OpenKE 有一个工具用于导入数据: :py:class:`pybind11_ke.data.GraphDataLoader`。
"""

from pybind11_ke.data import CompGCNSampler, CompGCNTestSampler, GraphDataLoader
from pybind11_ke.module.model import CompGCN
from pybind11_ke.module.loss import Cross_Entropy_Loss
from pybind11_ke.module.strategy import CompGCNSampling
from pybind11_ke.config import Trainer, GraphTester

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.GraphDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = GraphDataLoader(
	in_path = "../../benchmarks/FB15K237/",
	batch_size = 2048,
	test_batch_size = 256,
	num_workers = 16,
	test = True,
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
# :py:class:`pybind11_ke.module.model.CompGCN`，这是一种在图卷积网络中整合多关系信息的新框架，
# 它利用知识图谱嵌入技术中的各种组合操作，将实体和关系共同嵌入到图中。

# define the model
compgcn = CompGCN(
	ent_tol = dataloader.train_sampler.ent_tol,
	rel_tol = dataloader.train_sampler.rel_tol,
	dim = 100
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``CompGCN`` :cite:`CompGCN` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.Cross_Entropy_Loss`，
# :py:class:`pybind11_ke.module.strategy.CompGCNSampling` 对
# :py:class:`pybind11_ke.module.loss.Cross_Entropy_Loss` 进行了封装。

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
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.Trainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Trainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`pybind11_ke.config.GraphTester`，
# 使得训练器能够在训练过程中评估模型；:py:class:`pybind11_ke.config.GraphTester` 使用
# :py:class:`pybind11_ke.data.GraphDataLoader` 作为数据采样器。

# test the model
tester = GraphTester(model = compgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 2000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 50, log_interval = 50,
	save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
)
trainer.run()