"""
`CompGCN-FB15K237-single-gpu <single_gpu_compgcn_FB15K237.html>`_ ||
**CompGCN-FB15K237-single-gpu-wandb** ||
`CompGCN-FB15K237-single-gpu-hpo <single_gpu_compgcn_FB15K237_hpo.html>`_

CompGCN-FB15K237-single-gpu-wandb
=====================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2024

这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``CompGCN`` :cite:`CompGCN`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有一个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, CompGCNSampler, CompGCNTestSampler
from pybind11_ke.module.model import CompGCN
from pybind11_ke.module.loss import CompGCNLoss
from pybind11_ke.module.strategy import CompGCNSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="compgcn",
	config=dict(
		in_path = '../../benchmarks/FB15K237/',
		batch_size = 2048,
		test = True,
		test_batch_size = 256,
		num_workers = 16,
		dim = 100,
		use_gpu = True,
		device = 'cuda:0',
		prediction = "tail",
		epochs = 2000,
		lr = 0.0001,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/compgcn.pth'
	)
)

config = wandb_logger.config

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

dataloader = KGEDataLoader(
	in_path = config.in_path,
	batch_size = config.batch_size,
	test = config.test,
	test_batch_size = config.test_batch_size,
	num_workers = config.num_workers,
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
# :py:class:`pybind11_ke.module.model.CompGCN`，它提出于 2017 年，是第一个图神经网络模型，

# define the model
compgcn = CompGCN(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = config.dim
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``CompGCN`` :cite:`CompGCN` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.CompGCNLoss`，
# :py:class:`pybind11_ke.module.strategy.CompGCNSampling` 对
# :py:class:`pybind11_ke.module.loss.CompGCNLoss` 进行了封装。

# define the loss function
model = CompGCNSampling(
	model = compgcn,
	loss = CompGCNLoss(model = compgcn),
	ent_tol = dataloader.get_ent_tol()
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
tester = Tester(
    model = compgcn, data_loader = dataloader,
    use_gpu = config.use_gpu, device = config.device,
    prediction = config.prediction
)

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs, lr = config.lr, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
    log_interval = config.log_interval, save_interval = config.save_interval,
    save_path = config.save_path, use_wandb = True
)
trainer.run()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/CompGCN/single_gpu_compgcn_FB15K237_wandb.txt>`_ 下载。
# .. Note:: 上述代码的运行报告可以从 `此处 </zh-cn/latest/_static/pdfs/examples/CompGCN/CompGCN单卡训练示例（一）.pdf>`_ 下载。

######################################################################
# --------------
#