"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
**TransE-FB15K-multigpu** ||
`TransE-FB15K-multigpu-wandb <multigpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
`TransE-WN18RR-single-gpu-adv-wandb <single_gpu_transe_WN18_adv_sigmoidloss_wandb.html>`_

TransE-FB15K-multigpu
====================================================================

这一部分介绍如何用多个 GPU 在 ``FB15k`` 知识图谱上训练 ``TransE`` :cite:`TransE`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import accelerator_prepare
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
# train_dataloader = TrainDataLoader(
# 	in_path = "/home/luyanfeng/my_code/github/pybind11-OpenKE/benchmarks/FB15K/", 
# 	batch_size = 8192*4,
# 	neg_ent = 25,
# 	num_workers = 16,
# 	train_sampler = BernSampler,
# )

# dataloader for training
dataloader = KGEDataLoader(
    in_path = "../../benchmarks/FB15K/",
    batch_size = 8192,
    neg_ent = 25,
    test = True,
    test_batch_size = 256,
    num_workers = 16,
    train_sampler = BernSampler,
    test_sampler = TradTestSampler
)

# valid_dataloader = TestDataLoader(
# 	test_batch_size = 256,
# 	num_workers = 16,
# 	train_sampler = train_dataloader.train_sampler,
# 	test_sampler = TradTestSampler
# )

# test_dataloader = TestDataLoader(
# 	test_batch_size = 256,
# 	num_workers = 16,
# 	train_sampler = train_dataloader.train_sampler,
# 	test_sampler = TradTestSampler
# )

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.TransE`，它是最简单的平移模型。

# define the model
transe = TransE(
	ent_tol = dataloader.train_sampler.ent_tol,
	rel_tol = dataloader.train_sampler.rel_tol,
	dim = 50, 
	p_norm = 1, 
	norm_flag = True
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
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:func:`pybind11_ke.config.trainer_distributed_data_parallel` 函数，
# 进行并行训练，该函数必须由 ``if __name__ == '__main__'`` 保护。

dataloader, model, accelerator = accelerator_prepare(
    dataloader,
    model
)

# test the model
valider = Tester(model = transe, data_loader=dataloader)

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 3, lr = 0.01, accelerator = accelerator,
	tester = valider, test = True, valid_interval = 1,
	log_interval = 1, save_interval = 1,
	save_path = '../../checkpoint/transe.pth', delta = 0.01)
trainer.run()