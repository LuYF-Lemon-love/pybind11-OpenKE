"""
**TransE-FB15K-multigpu** ||
`TransE-WN18RR-adv <train_transe_WN18_adv_sigmoidloss.html>`_ ||
`TransH-FB15K237 <train_transh_FB15K237.html>`_ ||
`SimplE-WN18RR <train_simple_WN18RR.html>`_ ||
`RotatE-WN18RR <train_rotate_WN18RR_adv.html>`_

TransE-FB15K-multigpu
=========================

这一部分介绍如何用在 ``FB15k`` 知识图谱上训练 TransE。

TransE 原论文: `Translating Embeddings for Modeling Multi-relational Data <https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`__。

下面是 TransE 原论文给出的 FB15k 的模型超参数，是使用 :py:class:`torch.optim.SGD` 进行训练的。

============= =========== ========== ============ ===========
   向量维度       学习率     margin      距离函数     epochs
============= =========== ========== ============ ===========
     50           0.01        1           L1         1,000
============= =========== ========== ============ ===========

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.config import Trainer, Tester, trainer_distributed_data_parallel
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# 
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../../benchmarks/FB15K/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern = False,  
	neg_ent = 25,
	neg_rel = 0)

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
	ent_tot = train_dataloader.get_ent_tol(),
	rel_tot = train_dataloader.get_rel_tol(),
	dim = 50, 
	p_norm = 1, 
	norm_flag = True)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 TransE 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.MarginLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
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
# 或者使用 :py:func:`pybind11_ke.config.TrainerDataParallel.trainer_distributed_data_parallel` 函数进行并行训练，
# 该函数必须由 ``if __name__ == '__main__'`` 保护。

if __name__ == "__main__":

	trainer_distributed_data_parallel(model = model, data_loader = train_dataloader,
		train_times = 1000, alpha = 0.02, opt_method = "adam",
		test = True, valid_interval = 100, log_interval = 100, save_interval = 100,
		save_path = "../../checkpoint/transe.pth", type_constrain = False)

######################################################################
# --------------
#

######################################################################
# 评估模型
# -------------
# :py:class:`pybind11_ke.data.TestDataLoader` 包含 ``in_path`` 用于传递数据集目录。
# 与模型训练一样，pybind11-OpenKE 将评估模型包装成了 :py:class:`pybind11_ke.config.Tester`，
# 可以运行它的 :py:meth:`pybind11_ke.config.Tester.run_link_prediction` 函数进行链接预测。