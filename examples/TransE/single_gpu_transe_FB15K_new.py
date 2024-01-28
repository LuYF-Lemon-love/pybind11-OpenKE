"""
**TransE-FB15K-single-gpu** ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
`TransE-FB15K-multigpu <multigpu_transe_FB15K.html>`_ ||
`TransE-FB15K-multigpu-wandb <multigpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
`TransE-WN18RR-single-gpu-adv-wandb <single_gpu_transe_WN18_adv_sigmoidloss_wandb.html>`_

TransE-FB15K-single-gpu
====================================================================

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上训练 ``TransE`` :cite:`TransE`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.config import UniTrainer, GraphTester
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import Sampling
from pybind11_ke.data import UniDataLoader, UniSampler, TestSampler

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = UniDataLoader(
    in_path = "../../benchmarks/FB15K237/",
    batch_size = 4096,
    test_batch_size = 256,
    num_workers = 16,
    test = True,
    train_sampler = UniSampler,
    test_sampler = TestSampler
)

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
	norm_flag = True)

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
model = Sampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
	batch_size = 4096
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
tester = GraphTester(model = transe, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")

# train the model
trainer = UniTrainer(model = model, data_loader = dataloader.train_dataloader(),
    epochs = 2000, lr = 0.01, use_gpu = True, device = 'cuda:0',
    tester = tester, test = True, valid_interval = 50, log_interval = 50,
    save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
)
trainer.run()