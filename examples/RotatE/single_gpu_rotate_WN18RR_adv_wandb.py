"""
`RotatE-WN18RR-single-gpu-adv <single_gpu_rotate_WN18RR_adv.html>`_ ||
**RotatE-WN18RR-single-gpu-adv-wandb** ||
`RotatE-WN18RR-single-gpu-adv-hpo <single_gpu_rotate_WN18RR_adv_hpo.html>`_

RotatE-WN18RR-single-gpu-adv-wandb
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 15, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 15, 2024

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``RotatE`` :cite:`RotatE`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有 1 个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, UniSampler, TradTestSampler
from pybind11_ke.module.model import RotatE
from pybind11_ke.module.loss import SigmoidLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="RotatE-WN18RR",
	config=dict(
		in_path = '../../benchmarks/WN18RR/',
		batch_size = 2000,
		neg_ent = 64,
		test = True,
		test_batch_size = 10,
		num_workers = 16,
		dim = 1024,
		margin = 6.0,
		epsilon = 2.0,
		adv_temperature = 2,
		regul_rate = 0.0,
        use_tqdm = False,
		use_gpu = True,
		device = 'cuda:1',
		epochs = 6000,
		lr = 2e-5,
		opt_method = 'adam',
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/rotate.pth'
	)
)

config = wandb_logger.config

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = config.in_path, 
	batch_size = config.batch_size,
	neg_ent = config.neg_ent,
	test = config.test,
	test_batch_size = config.test_batch_size,
	num_workers = config.num_workers,
	train_sampler = UniSampler,
	test_sampler = TradTestSampler
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.RotatE`，它将实体表示成复数向量，关系建模为复数向量空间的旋转。

# define the model
rotate = RotatE(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = config.dim,
	margin = config.margin,
	epsilon = config.epsilon,
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了逻辑损失函数：:py:class:`pybind11_ke.module.loss.SigmoidLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.SigmoidLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = config.adv_temperature),
	regul_rate = config.regul_rate
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
tester = Tester(model = rotate, data_loader = dataloader, use_tqdm = config.use_tqdm,
                use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(), epochs = config.epochs,
	lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, use_wandb = True)
trainer.run()

# close your wandb run
wandb_logger.finish()

######################################################################
# .. figure:: /_static/images/examples/RotatE/RotatE-WN18RR-Loss.png
#      :align: center
#      :height: 300
#
#      训练过程中损失值的变化

######################################################################
# .. figure:: /_static/images/examples/RotatE/RotatE-WN18RR-MR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MR 的变化

######################################################################
# .. figure:: /_static/images/examples/RotatE/RotatE-WN18RR-MRR.png
#      :align: center
#      :height: 300
#
#      训练过程中 MRR 的变化

######################################################################
# .. figure:: /_static/images/examples/RotatE/RotatE-WN18RR-Hit.png
#      :align: center
#      :height: 300
#
#      训练过程中 Hits@3、Hits@3 和 Hits@10 的变化

######################################################################
# --------------
#