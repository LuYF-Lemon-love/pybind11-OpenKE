"""
**ANALOGY-WN18RR-single-gpu** ||
`ANALOGY-WN18RR-single-gpu-wandb <single_gpu_analogy_WN18RR_wandb.html>`_ ||
`ANALOGY-WN18RR-single-gpu-hpo <single_gpu_analogy_WN18RR_hpo.html>`_

ANALOGY-WN18RR-single-gpu
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 19, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 19, 2024

这一部分介绍如何用一个 GPU 在 ``WN18RR`` 知识图谱上训练 ``ANALOGY`` :cite:`ANALOGY`。

导入数据
-----------------
pybind11-OpenKE 有 1 个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader`。
"""

from pybind11_ke.data import KGEDataLoader
from pybind11_ke.module.model import Analogy
from pybind11_ke.module.loss import SoftplusLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.config import Trainer, Tester

######################################################################
# pybind11-OpenKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = "../../benchmarks/WN18RR/", 
	batch_size = 4096,
	neg_ent = 25,
	test = True, 
	test_batch_size = 10,
	num_workers = 16,
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.Analogy`，它是双线性模型的集大成者。

# define the model
analogy = Analogy(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = 200
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
	model = analogy, 
	loss = SoftplusLoss(), 
	regul_rate = 1.0
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
tester = Tester(model = analogy, data_loader = dataloader, use_tqdm = False,
                use_gpu = True, device = 'cuda:0')

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = 2000, lr = 0.5, opt_method = "adagrad", use_gpu = True, device = 'cuda:0',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100,
	save_path = '../../checkpoint/analogy.pth', delta = 0.01)
trainer.run()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/ANALOGY/single_gpu_analogy_WN18RR.txt>`_ 下载。

######################################################################
# --------------
#