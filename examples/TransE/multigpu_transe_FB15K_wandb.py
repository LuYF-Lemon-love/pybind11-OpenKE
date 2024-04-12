"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
`TransE-FB15K-multigpu <multigpu_transe_FB15K.html>`_ ||
**TransE-FB15K-multigpu-wandb** ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
`TransE-WN18RR-single-gpu-adv-wandb <single_gpu_transe_WN18_adv_sigmoidloss_wandb.html>`_

TransE-FB15K-multigpu-wandb
====================================================================

这一部分介绍如何用多个 GPU 在 ``FB15k`` 知识图谱上训练 ``TransE`` :cite:`TransE`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
from pybind11_ke.module.model import TransE
from pybind11_ke.config import trainer_distributed_data_parallel
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="TransE-FB15K-multi",
	config=dict(
		in_path = "../../benchmarks/FB15K/",
		batch_size = 1024,
		neg_ent = 25,
		test = True,
		test_batch_size = 256,
		num_workers = 16,
		dim = 50,
		p_norm = 1,
		norm_flag = True,
		margin = 1.0,
		epochs = 1000,
		lr = 0.01,
		opt_method = "adam",
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transe.pth',
		delta = 0.01
	)
)

config = wandb_logger.config

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = config.in_path, 
	batch_size = config.batch_size,
	neg_ent = config.neg_ent,
	test = config.test,
	test_batch_size = config.test_batch_size,
	num_workers = config.num_workers,
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
	ent_tol = dataloader.train_sampler.ent_tol,
	rel_tol = dataloader.train_sampler.rel_tol,
	dim = config.dim, 
	p_norm = config.p_norm, 
	norm_flag = config.norm_flag)

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
	loss = MarginLoss(margin = config.margin)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:func:`pybind11_ke.config.trainer_distributed_data_parallel` 函数，
# 进行并行训练，该函数必须由 ``if __name__ == '__main__'`` 保护。

if __name__ == "__main__":

	print("Start parallel training...")

	trainer_distributed_data_parallel(model = model,
		epochs = config.epochs, lr = config.lr, opt_method = config.opt_method,
		valid_interval = config.valid_interval, log_interval = config.log_interval,
		save_interval = config.save_interval, save_path = config.save_path)
	
	# close your wandb run
	wandb_logger.finish()