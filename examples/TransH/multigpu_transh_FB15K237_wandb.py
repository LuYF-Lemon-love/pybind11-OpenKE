"""
`TransH-FB15K237-single-gpu <single_gpu_transh_FB15K237.html>`_ ||
`TransH-FB15K237-single-gpu-wandb <single_gpu_transh_FB15K237_wandb.html>`_ ||
`TransH-FB15K237-single-gpu-hpo <single_gpu_transh_FB15K237_hpo.html>`_ ||
`TransH-FB15K237-multigpu <multigpu_transh_FB15K237.html>`_ ||
**TransH-FB15K-multigpu-wandb**

TransH-FB15K-multigpu-wandb
=====================================================

这一部分介绍如何用多个 GPU 在 FB15K237 知识图谱上训练 ``TransH`` :cite:`TransH`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。
"""

from pybind11_ke.utils import WandbLogger
from pybind11_ke.config import trainer_distributed_data_parallel
from pybind11_ke.module.model import TransH
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader

######################################################################
# 首先初始化 :py:class:`pybind11_ke.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="transh",
	config=dict(
		in_path = "../../benchmarks/FB15K237/",
		nbatches = 100,
		threads = 8,
		sampling_mode = "normal",
		bern = True,
		neg_ent = 25,
		neg_rel = 0,
		dim = 200,
		p_norm = 1,
		norm_flag = True,
		margin = 4.0,
		use_gpu = True,
		device = 'cuda:1',
		epochs = 1000,
		lr = 0.5,
		opt_method = "adam",
		test = True,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transh.pth',
		type_constrain = True
	)
)

config = wandb_logger.config

######################################################################
# pybind11-KE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`pybind11_ke.data.TrainDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = config.in_path, 
	nbatches = config.nbatches,
	threads = config.threads, 
	sampling_mode = config.sampling_mode, 
	bern = config.bern,  
	neg_ent = config.neg_ent,
	neg_rel = config.neg_rel)

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
	dim = config.dim, 
	p_norm = config.p_norm,
	norm_flag = config.norm_flag)

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
	loss = MarginLoss(margin = config.margin),
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
		epochs = config.epochs, lr = config.lr, opt_method = config.opt_method,
		test = config.test, valid_interval = config.valid_interval, log_interval = config.log_interval,
		save_interval = config.save_interval, save_path = config.save_path,
		type_constrain = True, use_wandb = True)

	# close your wandb run
	wandb_logger.finish()