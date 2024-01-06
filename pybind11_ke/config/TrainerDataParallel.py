# coding:utf-8
#
# pybind11_ke/config/TrainerDataParallel.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 5, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 6, 2023
#
# 该脚本定义了并行训练循环函数.

"""
trainer_distributed_data_parallel - 并行训练循环函数。
"""

import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from .Tester import Tester
from .Trainer import Trainer
from ..data import TrainDataLoader, TestDataLoader
from ..module.strategy import NegativeSampling

def ddp_setup(rank: int, world_size: int):

	"""
	构建进程组。在 Windows 上， :py:mod:`torch.distributed` 仅支持 `Gloo` 后端。

	:param rank: 进程的唯一标识符
	:type rank: int
	:param world_size: 进程的总数
	:type world_size: int
	"""
	
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	init_process_group(backend="gloo", rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)

def train(
	rank: int,
	world_size: int,
	model: NegativeSampling,
	data_loader: TrainDataLoader,
	epochs: int,
	lr: float,
	opt_method: str,
	test: bool,
	valid_interval: int | None,
	log_interval: int | None,
	save_interval: int | None,
	save_path: str | None,
	use_early_stopping: bool,
	metric: str,
	patience: int,
	delta: float,
	valid_file: str,
	test_file: str,
	type_constrain: bool,
	use_wandb: bool):

	"""进程函数。

	:param gpu_id: 第几个 gpu
	:type gpu_id: int
	:param world_size: 进程的总数
	:type world_size: int
	:param model: 包装 KGE 模型的训练策略类
	:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling`
	:param data_loader: TrainDataLoader
	:type data_loader: :py:class:`pybind11_ke.data.TrainDataLoader`
	:param epochs: 训练轮次数
	:type epochs: int
	:param lr: 学习率
	:type lr: float
	:param opt_method: 优化器: ``Adam`` or ``adam``, ``SGD`` or ``sgd``
	:type opt_method: str
	:param test: 是否在测试集上评估模型
	:type test: bool
	:param valid_interval: 训练几轮在验证集上评估一次模型
	:type valid_interval: int
	:param log_interval: 训练几轮输出一次日志
	:type log_interval: int
	:param save_interval: 训练几轮保存一次模型
	:type save_interval: int
	:param save_path: 模型保存的路径
	:type save_path: str
	:param use_early_stopping: 是否启用早停
	:type use_early_stopping: bool
	:param metric: 早停使用的验证指标，可选值：'mrr', 'hit1', 'hit3', 'hit10', 'mrTC', 'mrrTC', 'hit1TC', 'hit3TC', 'hit10TC'。
		'mrTC', 'mrrTC', 'hit1TC', 'hit3TC', 'hit10TC' 需要 :py:attr:`pybind11_ke.data.TestDataLoader.type_constrain` 为 True。
	:type metric: str
	:param patience: :py:attr:`pybind11_ke.utils.EarlyStopping.patience` 参数，上次验证得分改善后等待多长时间。
	:type patience: int
	:param delta: :py:attr:`pybind11_ke.utils.EarlyStopping.delta` 参数，监测数量的最小变化才符合改进条件。
	:type delta: float
	:param valid_file: valid2id.txt
	:type valid_file: str
	:param test_file: test2id.txt
	:type test_file: str
	:param type_constrain: 是否用 type_constrain.txt 进行负采样
	:type type_constrain: bool
	:param use_wandb: 是否启用 wandb 进行日志输出
	:type use_wandb: bool
	"""
	
	ddp_setup(rank, world_size)
	tester = None
	if test:
		test_dataloader = TestDataLoader(in_path = data_loader.in_path, ent_file = data_loader.ent_file,
			rel_file = data_loader.rel_file, train_file = data_loader.train_file, valid_file = valid_file,
			test_file = test_file, type_constrain = type_constrain)
		tester = Tester(model = model.model, data_loader = test_dataloader)
	trainer = Trainer(
		model=model,
		data_loader=data_loader,
		epochs=epochs,
		lr=lr,
		opt_method=opt_method,
		tester=tester,
		test=test,
		valid_interval=valid_interval,
		log_interval=log_interval,
		save_interval=save_interval,
		save_path=save_path,
		use_early_stopping=use_early_stopping,
		metric=metric,
		patience=patience,
		delta=delta,
		use_wandb=use_wandb,
		gpu_id=rank)
	try:
		trainer.run()
	except RuntimeError as err:
		print("RuntimeError:", err)
		print(f"[GPU0] Due to the lack of improvement in validation scores, an early stop has been made, and GPU{rank} also needs to stop model training.")
	finally:
		destroy_process_group()
	
def trainer_distributed_data_parallel(
	model = None,
	data_loader: TrainDataLoader | None = None,
	epochs: int = 1000,
	lr: float = 0.5,
	opt_method: str = "Adam",
	test: bool = False,
	valid_interval: int | None = None,
	log_interval: int | None = None,
	save_interval: int | None = None,
	save_path: str | None = None,
	use_early_stopping: bool = True,
	metric: str = 'hit10',
	patience: int = 2,
	delta: float = 0,
	valid_file: str = "valid2id.txt",
	test_file: str = "test2id.txt",
	type_constrain: bool = True,
	use_wandb: bool = False):

	"""并行训练循环函数，用于生成单独子进程进行训练模型。
	
	:py:mod:`torch.multiprocessing` 是 Python 原生 ``multiprocessing`` 的一个 ``PyTorch`` 的包装器。
	``multiprocessing`` 的生成进程函数必须由 ``if __name__ == '__main__'`` 保护。
	有效的 batch size 是 :py:attr:`pybind11_ke.data.TrainDataLoader.batch_size` * ``nprocs``。

	例子::

		from pybind11_ke.config import trainer_distributed_data_parallel

		if __name__ == "__main__":
		
			trainer_distributed_data_parallel(model = model, data_loader = train_dataloader,
				epochs = 1000, lr = 0.02, opt_method = "adam",
				test = True, valid_interval = 100, log_interval = 100, save_interval = 100,
				save_path = "../../checkpoint/transe.pth", type_constrain = False)

	:param model: 包装 KGE 模型的训练策略类
	:type model: :py:class:`pybind11_ke.module.strategy.NegativeSampling`
	:param data_loader: TrainDataLoader
	:type data_loader: :py:class:`pybind11_ke.data.TrainDataLoader`
	:param epochs: 训练轮次数
	:type epochs: int
	:param lr: 学习率
	:type lr: float
	:param opt_method: 优化器: ``Adam`` or ``adam``, ``SGD`` or ``sgd``
	:type opt_method: str
	:param test: 是否在测试集上评估模型
	:type test: bool
	:param valid_interval: 训练几轮在验证集上评估一次模型
	:type valid_interval: int
	:param log_interval: 训练几轮输出一次日志
	:type log_interval: int
	:param save_interval: 训练几轮保存一次模型
	:type save_interval: int
	:param save_path: 模型保存的路径
	:type save_path: str
	:param use_early_stopping: 是否启用早停，需要 :py:attr:`tester` 和 :py:attr:`save_path` 不为空
	:type use_early_stopping: bool
	:param metric: 早停使用的验证指标，可选值：'mrr', 'hit1', 'hit3', 'hit10', 'mrTC', 'mrrTC', 'hit1TC', 'hit3TC', 'hit10TC'。
		'mrTC', 'mrrTC', 'hit1TC', 'hit3TC', 'hit10TC' 需要 :py:attr:`pybind11_ke.data.TestDataLoader.type_constrain` 为 True。默认值：'hit10'
	:type metric: str
	:param patience: :py:attr:`pybind11_ke.utils.EarlyStopping.patience` 参数，上次验证得分改善后等待多长时间。默认值：2
	:type patience: int
	:param delta: :py:attr:`pybind11_ke.utils.EarlyStopping.delta` 参数，监测数量的最小变化才符合改进条件。默认值：0
	:type delta: float
	:param valid_file: valid2id.txt
	:type valid_file: str
	:param test_file: test2id.txt
	:type test_file: str
	:param type_constrain: 是否用 type_constrain.txt 进行负采样
	:type type_constrain: bool
	:param use_wandb: 是否启用 wandb 进行日志输出
	:type use_wandb: bool
	"""
	
	world_size = torch.cuda.device_count()
	mp.spawn(train, args = (world_size, model, data_loader, epochs, lr, opt_method,
							test, valid_interval, log_interval, save_interval, save_path,
							use_early_stopping, metric, patience, delta,
							valid_file, test_file, type_constrain, use_wandb),
				nprocs = world_size)