"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
`TransE-FB15K-single-gpu-wandb <single_gpu_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
`TransE-FB15K-accelerate <accelerate_transe_FB15K.html>`_ ||
`TransE-FB15K-accelerate-wandb <accelerate_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
**TransE-WN18RR-single-gpu-adv-wandb**

TransE-WN18RR-single-gpu-adv-wandb
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 11, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 11, 2024

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``TransE`` :cite:`TransE`，应用 ``RotatE`` :cite:`RotatE` 提出的自我对抗负采样损失函数进行模型训练，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。

"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.data import KGEDataLoader, UniSampler, TradTestSampler
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import SigmoidLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="TransE-WN18RR",
	config=dict(
		in_path = "../../benchmarks/WN18RR/",
		batch_size = 2000,
		neg_ent = 64,
		test = True,
		test_batch_size = 1,
		num_workers = 16,
		dim = 1024,
		p_norm = 1,
		norm_flag = False,
		margin = 6.0,
		adv_temperature = 1,
		regul_rate = 0.0,
		use_gpu = True,
		device = 'cuda:1',
		epochs = 3000,
		lr = 2e-5,
		opt_method = "adam",
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transe.pth'
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
# :py:class:`pybind11_ke.module.model.TransE`，它是最简单的平移模型。

# define the model
transe = TransE(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = config.dim, 
	p_norm = config.p_norm,
	norm_flag = config.norm_flag,
	margin = config.margin)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了逻辑损失函数：:py:class:`pybind11_ke.module.loss.SigmoidLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.SigmoidLoss` 进行了封装，加入权重衰减等额外项。
# 除此之外，我们使用 ``adv_temperature`` 开启了 RotatE 提出的自我对抗负采样。

# define the loss function
model = NegativeSampling(
	model = transe, 
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
# 使得训练器能够在训练过程中评估模型。
	
# test the model
tester = Tester(model = transe, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs, lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, use_wandb = True)
trainer.run()

# close your wandb run
wandb_logger.finish()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/TransE/single_gpu_transe_WN18_adv_sigmoidloss_wandb.txt>`_ 下载。
# .. Note:: 上述代码的运行报告可以从 `此处 </zh-cn/latest/_static/pdfs/examples/TransE/TransE单卡训练示例（三）.pdf>`_ 下载。

######################################################################
# --------------
#
