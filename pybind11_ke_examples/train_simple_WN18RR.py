"""
`RESCAL-FB15K237 <train_rescal_FB15K237.html>`_ ||
`TransE-FB15K237 <train_transe_FB15K237.html>`_ ||
`TransH-FB15K237 <train_transh_FB15K237.html>`_ ||
`DistMult-WN18RR <train_distmult_WN18RR.html>`_ ||
`TransD-FB15K237 <train_transd_FB15K237.html>`_ ||
`HolE-WN18RR <train_hole_WN18RR.html>`_ ||
`ComplEx-WN18RR <train_complex_WN18RR.html>`_ ||
`Analogy-WN18RR <train_analogy_WN18RR.html>`_ ||
**SimplE-WN18RR** ||
`RotatE-WN18RR <train_rotate_WN18RR_adv.html>`_

SimplE-WN18RR
===================
这一部分介绍如何用在 WN18RR 知识图谱上训练 SimplE。

SimplE 原论文: `SimplE Embedding for Link Prediction in Knowledge Graphs <https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html>`__。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。

"""

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import SimplE
from pybind11_ke.module.loss import SoftplusLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# 
# :py:class:`pybind11_ke.data.TrainDataLoader` 和 :py:class:`pybind11_ke.data.TestDataLoader`
# 都包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/WN18RR/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("../benchmarks/WN18RR/", "link")

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.SimplE`，它是简单的双线性模型，
# 能够为头实体和尾实体学习不同的嵌入向量。

# define the model
simple = SimplE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
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
	model = simple, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:class:`pybind11_ke.config.Trainer`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Trainer.run` 函数进行模型学习。

# train the model
# train_times: 2000 -> 6000
trainer = Trainer(model = model, data_loader = train_dataloader,
                  train_times = 6000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
simple.save_checkpoint('../checkpoint/simple.ckpt')

######################################################################
# --------------
#

######################################################################
# 评估模型
# -------------
# 与模型训练一样，pybind11-OpenKE 将评估模型包装成了 :py:class:`pybind11_ke.config.Tester`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Tester.run_link_prediction` 函数进行链接预测。

# test the model
simple.load_checkpoint('../checkpoint/simple.ckpt')
tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
