"""
**RotatE-WN18RR-single-gpu-adv** ||
`RotatE-WN18RR-single-gpu-adv-wandb <single_gpu_rotate_WN18RR_adv_wandb.html>`_

RotatE-WN18RR-single-gpu-adv
====================================================================

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``RotatE`` :cite:`RotatE`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import RotatE
from pybind11_ke.module.loss import SigmoidLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = '../../benchmarks/WN18RR/', 
	batch_size = 2000,
	threads = 8,
	sampling_mode = 'cross',
	bern = False,
	neg_ent = 64,
	neg_rel = 0,
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
	ent_tot = train_dataloader.get_ent_tol(),
	rel_tot = train_dataloader.get_rel_tol(),
	dim = 1024,
	margin = 6.0,
	epsilon = 2.0,
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
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0,
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

# dataloader for test
test_dataloader = TestDataLoader(in_path = '../../benchmarks/WN18RR/')

# test the model
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, epochs = 6000,
	lr = 2e-5, opt_method = 'adam', use_gpu = True, device = 'cuda:1',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100,
	save_path = '../../checkpoint/rotate.pth', use_wandb = False)
trainer.run()

######################################################################
# --------------
#