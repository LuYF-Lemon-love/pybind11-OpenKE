"""
`TransR-FB15K237-single-gpu <single_gpu_transr_FB15K237.html>`_ ||
**TransR-FB15K237-single-gpu-wandb** ||
`TransR-FB15K237-single-gpu-hpo <single_gpu_transr_FB15K237_hpo.html>`_ ||
`TransR-FB15K237-multigpu <multigpu_transr_FB15K237.html>`_

TransR-FB15K237-single-gpu-wandb
=====================================================

这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``TransR`` :cite:`TransR`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, BernSampler, TradTestSampler
from pybind11_ke.module.model import TransE, TransR
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="TransR-FB15K237",
	config=dict(
		in_path = '../../benchmarks/FB15K237/',
		batch_size = 2048,
		neg_ent = 25,
		test = True,
		test_batch_size = 10,
		num_workers = 16,
		dim = 100,
		dim_e = 100,
		dim_r = 100,
		p_norm = 1,
		norm_flag = True,
		rand_init = False,
		margin_e = 5.0,
		margin_r = 4.0,
		epochs_e = 1,
		lr_e = 0.5,
		opt_method = "sgd",
		use_gpu = True,
		device = 'cuda:0',
		epochs_r = 1000,
		lr_r = 1.0,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transr.pth'
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

######################################################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们首先导入
# :py:class:`pybind11_ke.module.model.TransE`，它是最简单的平移模型，
# 因为为了避免过拟合，:py:class:`pybind11_ke.module.model.TransR` 实体和关系的嵌入向量初始化为
# :py:class:`pybind11_ke.module.model.TransE` 的结果。

# define the transe
transe = TransE(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = config.dim, 
	p_norm = config.p_norm, 
	norm_flag = config.norm_flag)

######################################################################
# 下面导入 :py:class:`pybind11_ke.module.model.TransR` 模型，
# 是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。

transr = TransR(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim_e = config.dim_e,
	dim_r = config.dim_r,
	p_norm = config.p_norm, 
	norm_flag = config.norm_flag,
	rand_init = config.rand_init)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``TransE`` :cite:`TransE` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.MarginLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = config.margin_e)
)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = config.margin_r)
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

# pretrain transe
trainer = Trainer(model = model_e, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs_e, lr = config.lr_e, opt_method = config.opt_method,
	use_gpu = config.use_gpu, device = config.device)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("../../checkpoint/transr_transe.json")

# test the transr
tester = Tester(model = transr, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs_r, lr = config.lr_r, opt_method = config.opt_method,
	use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = True, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, use_wandb = True)
trainer.run()

# test the model
transr.load_checkpoint('../../checkpoint/transr.pth')
tester.set_sampling_mode("link_test")
tester.run_link_prediction()

######################################################################
# .. figure:: /_static/images/examples/TransR/TransR-FB15K237-Loss.png
#      :align: center
#      :height: 300
#
#      训练过程中损失值的变化

######################################################################
# .. figure:: /_static/images/examples/TransR/TransR-FB15K237-MR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MR 的变化

######################################################################
# .. figure:: /_static/images/examples/TransR/TransR-FB15K237-MRR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MRR 的变化

######################################################################
# .. figure:: /_static/images/examples/TransR/TransR-FB15K237-Hit.png
#      :align: center
#      :height: 300
#
#      训练过程中 Hits@3、Hits@3 和 Hits@10 的变化

######################################################################
# --------------
#