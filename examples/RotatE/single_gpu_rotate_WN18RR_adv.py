"""
**RotatE-WN18RR-single-gpu-adv** ||
`RotatE-WN18RR-single-gpu-adv-wandb <single_gpu_rotate_WN18RR_adv_wandb.html>`_ ||
`RotatE-WN18RR-single-gpu-adv-hpo <single_gpu_rotate_WN18RR_adv_hpo.html>`_

RotatE-WN18RR-single-gpu-adv
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 15, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 15, 2024

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``RotatE`` :cite:`RotatE`。

导入数据
-----------------
pybind11-OpenKE 有 1 个工具用于导入数据: :py:class:`pybind11_ke.data.KGEDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader, UniSampler, TradTestSampler
from pybind11_ke.module.model import RotatE
from pybind11_ke.module.loss import SigmoidLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = '../../benchmarks/WN18RR/', 
	batch_size = 2000,
	neg_ent = 64,
	test = True,
	test_batch_size = 10,
	num_workers = 16,
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
# 使得训练器能够在训练过程中评估模型。

# test the model
tester = Tester(model = rotate, data_loader = dataloader, use_tqdm = False,
                use_gpu = True, device = 'cuda:1')

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(), epochs = 6000,
	lr = 2e-5, opt_method = 'adam', use_gpu = True, device = 'cuda:1',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100,
	save_path = '../../checkpoint/rotate.pth', use_wandb = False)
trainer.run()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/RotatE/single_gpu_rotate_WN18RR_adv.txt>`_ 下载。

######################################################################
# --------------
#