"""
**TransE-FB15K-single-gpu**

DistMult-WN18RR-single-gpu-wandb
====================================================================

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``DistMult`` :cite:`DistMult`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import DistMult
from pybind11_ke.module.loss import SoftplusLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="distMult",
	config=dict(
		in_path = '../../benchmarks/WN18RR/',
		nbatches = 100,
		threads = 8,
		sampling_mode = 'normal',
		bern = True,
		neg_ent = 25,
		neg_rel = 0,
		dim = 200,
		regul_rate = 1.0,
		use_gpu = True,
		device = 'cuda:1',
		epochs = 2000,
		lr = 0.5,
		opt_method = 'adagrad',
		test = True,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transe.pth'
	)
)

config = wandb_logger.config

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 都包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = config.in_path, 
	nbatches = config.nbatches,
	threads = config.threads, 
	sampling_mode = config.sampling_mode, 
	bern = config.bern,
	neg_ent = config.neg_ent,
	neg_rel = config.neg_rel
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.DistMult`，它是最简单的双线性模型。

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tol(),
	rel_tot = train_dataloader.get_rel_tol(),
	dim = config.dim
)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了逻辑损失函数：:py:class:`pybind11_ke.module.loss.SoftplusLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.SoftplusLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = config.regul_rate
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.Trainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Trainer.run` 函数进行模型学习。

# dataloader for test
test_dataloader = TestDataLoader(in_path = config.in_path)

# test the model
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, epochs = config.epochs,
	lr = config.lr, opt_method = config.opt_method, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, use_wandb = True)
trainer.run()

# close your wandb run
wandb_logger.finish()

######################################################################
# --------------
#