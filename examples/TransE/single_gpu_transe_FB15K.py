"""
**TransE-FB15K-single-gpu** ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
`TransE-FB15K-accelerate <accelerate_transe_FB15K.html>`_ ||
`TransE-FB15K-accelerate-wandb <accelerate_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
`TransE-WN18RR-single-gpu-adv-wandb <single_gpu_transe_WN18_adv_sigmoidloss_wandb.html>`_

TransE-FB15K-single-gpu
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2024

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上训练 ``TransE`` :cite:`TransE`。

导入数据
-----------------
pybind11-OpenKE 有 1 个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

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
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
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
	loss = MarginLoss(margin = 1.0),
	regul_rate = 0.01
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
tester = Tester(model = transe, data_loader = dataloader, use_gpu = True, device = 'cuda:0')

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100,
	save_path = '../../checkpoint/transe.pth', delta = 0.01)
# for linux
# trainer.run()

# for windows
if __name__ == '__main__':
	trainer.run()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/TransE/single_gpu_transe_FB15K.txt>`_ 下载。

######################################################################
# --------------
#