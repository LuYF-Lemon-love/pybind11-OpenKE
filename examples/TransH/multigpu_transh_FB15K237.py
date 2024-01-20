"""
`TransH-FB15K237-single-gpu <single_gpu_transh_FB15K237.html>`_ ||
`TransH-FB15K237-single-gpu-wandb <single_gpu_transh_FB15K237_wandb.html>`_ ||
**TransH-FB15K237-multigpu** ||
`TransH-FB15K-multigpu-wandb <multigpu_transh_FB15K237_wandb.html>`_

TransH-FB15K237-multigpu
=====================================================
这一部分介绍如何用多个 GPU 在 FB15K237 知识图谱上训练 ``TransH`` :cite:`TransH`。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.config import trainer_distributed_data_parallel
from pybind11_ke.module.model import TransH
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
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

################################
# 导入模型
# ------------------
# pybind11-OpenKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`pybind11_ke.module.model.TransH`，它提出于 2014 年，是第二个平移模型，
# 将关系建模为超平面上的平移操作。

# define the model
transh = TransH(
	ent_tol = train_dataloader.get_ent_tol(),
	rel_tol = train_dataloader.get_rel_tol(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``TransE`` :cite:`TransE` 原论文使用的损失函数：:py:class:`pybind11_ke.module.loss.MarginLoss`，
# :py:class:`pybind11_ke.module.strategy.NegativeSampling` 对
# :py:class:`pybind11_ke.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# pybind11-OpenKE 将训练循环包装成了 :py:func:`pybind11_ke.config.trainer_distributed_data_parallel` 函数，
# 进行并行训练，该函数必须由 ``if __name__ == '__main__'`` 保护。

if __name__ == "__main__":

	print("Start parallel training...")

	trainer_distributed_data_parallel(model = model, data_loader = train_dataloader,
		epochs = 1000, lr = 0.5, opt_method = "adam",
		test = True, valid_interval = 100, log_interval = 100, save_interval = 100,
		save_path = "../../checkpoint/transh.pth", type_constrain = True)