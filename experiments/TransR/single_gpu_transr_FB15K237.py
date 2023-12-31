"""
**TransR-FB15K237-single-gpu** ||
`TransR-FB15K237-multigpu <multigpu_transr_FB15K237.html>`_

TransH-FB15K237-single-gpu
=====================================================
这一部分介绍如何用一个 GPU 在 FB15K237 知识图谱上训练 ``TransR`` :cite:`TransR`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。

"""

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import TransE, TransR
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# 
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../../benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern = True, 
	neg_ent = 25,
	neg_rel = 0)

######################################################################
# --------------
#

######################################################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们首先导入
# :py:class:`pybind11_ke.module.model.TransE`，它是最简单的平移模型，
# 因为为了避免过拟合，:py:class:`pybind11_ke.module.model.TransR` 实体和关系的嵌入向量初始化为
# :py:class:`pybind11_ke.module.model.TransE` 的结果

# define the transe
transe = TransE(
	ent_tot = train_dataloader.get_ent_tol(),
	rel_tot = train_dataloader.get_rel_tol(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True)

######################################################################
# 下面导入 :py:class:`pybind11_ke.module.model.TransR` 模型，
# 是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。

transr = TransR(
	ent_tot = train_dataloader.get_ent_tol(),
	rel_tot = train_dataloader.get_rel_tol(),
	dim_e = 100,
	dim_r = 100,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

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
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
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

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader,
	epochs = 1, lr = 0.5, use_gpu = True, device = 'cuda:1')
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("../../checkpoint/transr_transe.json")

# dataloader for test
test_dataloader = TestDataLoader("../../benchmarks/FB15K237/")

# test the transr
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader,
	epochs = 1000, lr = 1.0, use_gpu = True, device = 'cuda:1',
	tester = tester, test = True, valid_interval = 100,
	log_interval = 100, save_interval = 100, save_path = '../../checkpoint/transr.pth')
trainer.run()
