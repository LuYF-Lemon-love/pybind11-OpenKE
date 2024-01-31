"""
`TransH-FB15K237-single-gpu <single_gpu_transh_FB15K237.html>`_ ||
**TransH-FB15K237-single-gpu-wandb** ||
`TransH-FB15K237-single-gpu-hpo <single_gpu_transh_FB15K237_hpo.html>`_ ||
`TransH-FB15K237-multigpu <multigpu_transh_FB15K237.html>`_ ||
`TransH-FB15K-multigpu-wandb <multigpu_transh_FB15K237_wandb.html>`_

TransH-FB15K237-single-gpu-wandb
====================================================================

这一部分介绍如何用一个 GPU 在 ``FB15k237`` 知识图谱上训练 ``TransH`` :cite:`TransH`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
from pybind11_ke.module.model import TransH
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import TradTrainer, Tester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="TransH-FB15K237",
	config=dict(
		in_path = '../../benchmarks/FB15K237/',
		batch_size = 4096,
		neg_ent = 256,
		test = True,
		test_batch_size = 10,
		num_workers = 16,
		dim = 200,
		p_norm = 1,
		norm_flag = True,
		margin = 4.0,
		use_gpu = True,
		device = 'cuda:1',
		epochs = 1000,
		lr = 1.0e-05,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transh.pth'
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
# :py:class:`pybind11_ke.module.model.TransH`，它提出于 2014 年，是第二个平移模型，
# 将关系建模为超平面上的平移操作。

# define the model
transh = TransH(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
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
	model = transh, 
	loss = MarginLoss(margin = config.margin)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.TradTrainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.TradTrainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`pybind11_ke.config.Tester`，
# 使得训练器能够在训练过程中评估模型。
	
# test the model
tester = Tester(model = transh, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = TradTrainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs, lr = config.lr, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, use_wandb = True)
trainer.run()

# close your wandb run
wandb_logger.finish()

######################################################################
# .. figure:: /_static/images/examples/TransH/TransH-FB15K237-Loss.png
#      :align: center
#      :height: 300
#
#      训练过程中损失值的变化

######################################################################
# .. figure:: /_static/images/examples/TransH/TransH-FB15K237-MR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MR 的变化

######################################################################
# .. figure:: /_static/images/examples/TransH/TransH-FB15K237-MRR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MRR 的变化

######################################################################
# .. figure:: /_static/images/examples/TransH/TransH-FB15K237-Hit.png
#      :align: center
#      :height: 300
#
#      训练过程中 Hits@3、Hits@3 和 Hits@10 的变化

######################################################################
# --------------
#