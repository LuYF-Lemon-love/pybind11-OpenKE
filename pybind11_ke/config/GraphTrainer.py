# coding:utf-8
#
# pybind11_ke/config/GraphTrainer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 16, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 18, 2023
#
# 该脚本定义了 R-GCN 训练循环类.

"""
GraphTrainer - 训练循环类。
"""

import dgl
import typing
import torch
from .Trainer import Trainer
from .GraphTester import GraphTester
from ..module.strategy import RGCNSampling, CompGCNSampling
from torch.utils.data import DataLoader
from typing_extensions import override

class GraphTrainer(Trainer):

	"""
	主要用于 ``R-GCN`` :cite:`R-GCN` 和 ``CompGCN`` :cite:`CompGCN` 的训练。

	例子::

		from pybind11_ke.data import CompGCNSampler, CompGCNTestSampler, GraphDataLoader
		from pybind11_ke.module.model import CompGCN
		from pybind11_ke.module.loss import Cross_Entropy_Loss
		from pybind11_ke.module.strategy import CompGCNSampling
		from pybind11_ke.config import GraphTrainer, GraphTester
		
		dataloader = GraphDataLoader(
			in_path = "../../benchmarks/FB15K237/",
			batch_size = 2048,
			test_batch_size = 256,
			num_workers = 16,
			train_sampler = CompGCNSampler,
			test_sampler = CompGCNTestSampler
		)
		
		# define the model
		compgcn = CompGCN(
			ent_tol = dataloader.train_sampler.ent_tol,
			rel_tol = dataloader.train_sampler.rel_tol,
			dim = 100
		)
		
		# define the loss function
		model = CompGCNSampling(
			model = compgcn,
			loss = Cross_Entropy_Loss(model = compgcn),
			ent_tol = dataloader.train_sampler.ent_tol
		)
		
		# test the model
		tester = GraphTester(model = compgcn, data_loader = dataloader, use_gpu = True, device = 'cuda:0', prediction = "tail")
		
		# train the model
		trainer = GraphTrainer(model = model, data_loader = dataloader.train_dataloader(),
			epochs = 2000, lr = 0.0001, use_gpu = True, device = 'cuda:0',
			tester = tester, test = True, valid_interval = 50, log_interval = 50,
			save_interval = 50, save_path = '../../checkpoint/compgcn.pth'
		)
		trainer.run()
	"""

	def __init__(
		self,
		model: RGCNSampling | CompGCNSampling | None = None,
		data_loader: typing.Union[DataLoader, None] = None,
		epochs: int = 1000,
		lr: float = 0.5,
		opt_method: str = "Adam",
		use_gpu: bool = True,
		device: str = "cuda:0",
		tester: GraphTester | None = None,
		test: bool = False,
		valid_interval: int | None = None,
		log_interval: int | None = None,
		save_interval: int | None = None,
		save_path: str | None = None,
		use_early_stopping: bool = True,
		metric: str = 'hit10',
		patience: int = 2,
		delta: float = 0,
		use_wandb: bool = False,
		gpu_id: int | None = None):

		"""创建 GraphTrainer 对象。

		:param model: 包装 KGE 模型的训练策略类
		:type model: :py:class:`pybind11_ke.module.strategy.RGCNSampling` or :py:class:`pybind11_ke.module.strategy.CompGCNSampling`
		:param data_loader: DataLoader
		:type data_loader: torch.utils.data.DataLoader
		:param epochs: 训练轮次数
		:type epochs: int
		:param lr: 学习率
		:type lr: float
		:param opt_method: 优化器: 'Adam' or 'adam', 'Adagrad' or 'adagrad', 'SGD' or 'sgd'
		:type opt_method: str
		:param use_gpu: 是否使用 gpu
		:type use_gpu: bool
		:param device: 使用哪个 gpu
		:type device: str
		:param tester: 用于模型评估的验证模型类
		:type tester: :py:class:`pybind11_ke.config.Tester`
		:param test: 是否在测试集上评估模型, :py:attr:`tester` 不为空
		:type test: bool
		:param valid_interval: 训练几轮在验证集上评估一次模型, :py:attr:`tester` 不为空
		:type valid_interval: int
		:param log_interval: 训练几轮输出一次日志
		:type log_interval: int
		:param save_interval: 训练几轮保存一次模型
		:type save_interval: int
		:param save_path: 模型保存的路径
		:type save_path: str
		:param use_early_stopping: 是否启用早停，需要 :py:attr:`tester` 和 :py:attr:`save_path` 不为空
		:type use_early_stopping: bool
		:param metric: 早停使用的验证指标，可选值：'mr', 'mrr', 'hit1', 'hit3', 'hit10'。默认值：'hit10'
		:type metric: str
		:param patience: :py:attr:`pybind11_ke.utils.EarlyStopping.patience` 参数，上次验证得分改善后等待多长时间。默认值：2
		:type patience: int
		:param delta: :py:attr:`pybind11_ke.utils.EarlyStopping.delta` 参数，监测数量的最小变化才符合改进条件。默认值：0
		:type delta: float
		:param use_wandb: 是否启用 wandb 进行日志输出
		:type use_wandb: bool
		:param gpu_id: 第几个 gpu，用于并行训练
		:type gpu_id: int
		"""

		super(GraphTrainer, self).__init__(
			model=model,
			data_loader=data_loader,
			epochs=epochs,
			lr=lr,
			opt_method=opt_method,
			use_gpu=use_gpu,
			device=device,
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
			gpu_id=gpu_id
        )

	@override
	def train_one_step(
		self,
		data: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]) -> float:

		"""根据 :py:attr:`data_loader` 生成的 1 批次（batch） ``data`` 将
		模型训练 1 步。

		:param data: :py:attr:`data_loader` 利用 :py:meth:`pybind11_ke.data.GraphSampler.sampling` 函数生成的数据
		:type data: dict[str, typing.Union[dgl.DGLGraph , torch.Tensor]]
		:returns: 损失值
		:rtype: float
		"""

		self.optimizer.zero_grad()
		data = {key : self.to_var(value) for key, value in data.items()}
		loss = self.model(data)
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	@override
	def to_var(
		self,
		x: torch.Tensor) -> torch.Tensor:

		"""将 ``x`` 转移到对应的设备上。

		:param x: 数据
		:type x: torch.Tensor
		:returns: 张量
		:rtype: torch.Tensor
		"""

		if self.gpu_id is not None:
			return x.to(self.gpu_id)
		elif self.use_gpu:
			return x.to(self.device)
		else:
			return x

def get_graph_trainer_hpo_config() -> dict[str, dict[str, typing.Any]]:

	"""返回 :py:class:`GraphTrainer` 的默认超参数优化配置。
	
	默认配置为::
	
		parameters_dict = {
			'trainer': {
				'value': 'GraphTrainer'
			},
			'epochs': {
				'value': 10000
			},
			'lr': {
				'distribution': 'uniform',
				'min': 0,
				'max': 1.0
			},
			'opt_method': {
				'values': ['adam', 'adagrad', 'sgd']
			},
			'valid_interval': {
				'value': 50
			},
			'log_interval': {
				'value': 50
			},
			'save_path': {
				'value': './'
			},
			'use_early_stopping': {
				'value': True
			},
			'metric': {
				'value': 'hit10'
			},
			'patience': {
				'value': 2
			},
			'delta': {
				'value': 0
			},
		}

	:returns: :py:class:`GraphTrainer` 的默认超参数优化配置
	:rtype: dict[str, dict[str, typing.Any]]	
	"""

	parameters_dict = {
		'trainer': {
			'value': 'GraphTrainer'
		},
		'epochs': {
			'value': 10000
		},
		'lr': {
			'distribution': 'uniform',
			'min': 0,
			'max': 1.0
		},
		'opt_method': {
			'values': ['adam', 'adagrad', 'sgd']
		},
		'valid_interval': {
			'value': 50
		},
		'log_interval': {
			'value': 50
		},
		'save_path': {
			'value': './'
		},
		'use_early_stopping': {
			'value': True
		},
		'metric': {
			'value': 'hit10'
		},
		'patience': {
			'value': 2
		},
		'delta': {
			'value': 0
		},
	}
		
	return parameters_dict