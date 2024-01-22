# coding:utf-8
#
# pybind11_ke/config/HPOTrainer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 2, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 5, 2023
#
# 该脚本定义了并行训练循环函数.

"""
hpo_train - 超参数优化训练循环函数。
"""

import wandb
import typing
from ..data import TrainDataLoader, TestDataLoader
from ..utils import import_class
from ..module.strategy import NegativeSampling
from ..config import Trainer, Tester

def set_hpo_config(
	method: str = 'bayes',
	sweep_name: str = 'pybind11_ke_hpo',
	metric_name: str = 'val/hit10',
	metric_goal: str = 'maximize',
	train_data_loader_config: dict[str, dict[str, typing.Any]] | None = None,
	kge_config: dict[str, dict[str, typing.Any]] | None = None,
	loss_config: dict[str, dict[str, typing.Any]] | None = None,
	strategy_config: dict[str, dict[str, typing.Any]] | None = None,
	test_data_loader_config: dict[str, dict[str, typing.Any]] | None = None,
	tester_config: dict[str, dict[str, typing.Any]] | None = None,
	trainer_config: dict[str, dict[str, typing.Any]] | None = None) -> dict[str, dict[str, typing.Any]]:

	"""设置超参数优化范围。
	
	:param method: 超参数优化的方法，``grid`` 或 ``random`` 或 ``bayes``
	:type param: str
	:param sweep_name: 超参数优化 sweep 的名字
	:type sweep_name: str
	:param metric_name: 超参数优化的指标名字
	:type metric_name: str
	:param metric_goal: 超参数优化的指标目标，``maximize`` 或 ``minimize``
	:type metric_goal: str
	:param train_data_loader_config: :py:class:`pybind11_ke.data.TrainDataLoader` 的超参数优化配置
	:type train_data_loader_config: dict
	:param kge_config: :py:class:`pybind11_ke.module.model.Model` 的超参数优化配置
	:type kge_config: dict
	:param loss_config: :py:class:`pybind11_ke.module.loss.Loss` 的超参数优化配置
	:type loss_config: dict
	:param strategy_config: :py:class:`pybind11_ke.module.strategy.Strategy` 的超参数优化配置
	:type strategy_config: dict
	:param test_data_loader_config: :py:class:`pybind11_ke.data.TestDataLoader` 的超参数优化配置
	:type test_data_loader_config: dict
	:param tester_config: :py:class:`pybind11_ke.config.Tester` 的超参数优化配置
	:type tester_config: dict
	:param trainer_config: :py:class:`pybind11_ke.config.Trainer` 的超参数优化配置
	:type trainer_config: dict
	:returns: 超参数优化范围
	:rtype: dict
	"""

	sweep_config: dict[str, str] = {
		'method': method,
		'name': sweep_name
	}

	metric: dict[str, str] = {
		'name': metric_name,
		'goal': metric_goal
	}

	parameters_dict: dict[str, dict[str, typing.Any]] | None = {}
	parameters_dict.update(train_data_loader_config)
	parameters_dict.update(kge_config)
	parameters_dict.update(loss_config)
	parameters_dict.update(strategy_config)
	parameters_dict.update(test_data_loader_config)
	parameters_dict.update(tester_config)
	parameters_dict.update(trainer_config)

	sweep_config['metric'] = metric
	sweep_config['parameters'] = parameters_dict

	return sweep_config

def start_hpo_train(
	config: dict[str, dict[str, typing.Any]] | None = None,
	project: str = "pybind11-ke-sweeps",
	count: int = 2):

	"""开启超参数优化。
	
	:param config: wandb 的超参数优化配置。
	:type config: dict
	:param project: 项目名
	:type param: str
	:param count: 进行几次尝试。
	:type count: int
	"""

	wandb.login()

	sweep_id = wandb.sweep(config, project=project)

	wandb.agent(sweep_id, hpo_train, count=count)

def hpo_train(config: dict[str, typing.Any] | None = None):

	"""超参数优化训练循环函数。
	
	:param config: wandb 的项目配置如超参数。
	:type config: dict[str, typing.Any] | None
	"""
	
	with wandb.init(config = config):
		
		config = wandb.config

		# dataloader for training
		train_dataloader = TrainDataLoader(
		    in_path = config.in_path,
			ent_file = config.ent_file,
			rel_file = config.rel_file,
			train_file = config.train_file,
			batch_size = config.batch_size,
			threads = config.threads,
			sampling_mode = config.sampling_mode,
			bern = config.bern,
			neg_ent = config.neg_ent,
			neg_rel = config.neg_rel
		)

		# define the model
		model_class = import_class(f"pybind11_ke.module.model.{config.model}")
		if config.model in ["TransE", "TransH"]:
			kge_model = model_class(
			    ent_tol = train_dataloader.get_ent_tol(),
			    rel_tol = train_dataloader.get_rel_tol(),
			    dim = config.dim,
			    p_norm = config.p_norm,
			    norm_flag = config.norm_flag)
		elif config.model == "TransR":
			kge_model = model_class(
			    ent_tol = train_dataloader.get_ent_tol(),
			    rel_tol = train_dataloader.get_rel_tol(),
			    dim_e = config.dim_e,
				dim_r = config.dim_r,
			    p_norm = config.p_norm,
			    norm_flag = config.norm_flag,
				rand_init = config.rand_init)
		elif config.model == "TransD":
			kge_model = model_class(
			    ent_tol = train_dataloader.get_ent_tol(),
			    rel_tol = train_dataloader.get_rel_tol(),
			    dim_e = config.dim_e,
				dim_r = config.dim_r,
			    p_norm = config.p_norm,
			    norm_flag = config.norm_flag)
		elif config.model == "RotatE":
			kge_model = model_class(
			    ent_tol = train_dataloader.get_ent_tol(),
			    rel_tol = train_dataloader.get_rel_tol(),
			    dim = config.dim,
				margin = config.margin,
			    epsilon = config.epsilon)
		elif config.model in ["RESCAL", "DistMult", "HolE", "ComplEx"]:
			kge_model = model_class(
			    ent_tol = train_dataloader.get_ent_tol(),
			    rel_tol = train_dataloader.get_rel_tol(),
			    dim = config.dim)

		# define the loss function
		loss_class = import_class(f"pybind11_ke.module.loss.{config.loss}")
		if config.loss == 'MarginLoss':
			loss = loss_class(
				adv_temperature = config.adv_temperature,
				margin = config.margin)
		elif config.loss in ['SigmoidLoss', 'SoftplusLoss']:
			loss = loss_class(
				adv_temperature = config.adv_temperature)
		
		# define the strategt
		model = NegativeSampling(
		    model = kge_model,
		    loss = loss,
		    batch_size = train_dataloader.get_batch_size(),
			regul_rate = config.regul_rate,
			l3_regul_rate = config.l3_regul_rate
		)

		# dataloader for test
		test_dataloader = TestDataLoader(
			in_path = train_dataloader.in_path,
			ent_file = train_dataloader.ent_file,
			rel_file = train_dataloader.rel_file,
			train_file = train_dataloader.train_file,
			valid_file = config.valid_file,
			test_file = config.test_file,
			type_constrain = config.type_constrain
		)

		# test the model
		tester = Tester(
			model = kge_model,
			data_loader = test_dataloader,
			use_gpu = config.use_gpu,
			device = config.device
		)

		# # train the model
		trainer = Trainer(
			model = model,
			data_loader = train_dataloader,
		    epochs = config.epochs,
			lr = config.lr,
			opt_method = config.opt_method,
			use_gpu = config.use_gpu,
			device = config.device,
		    tester = tester,
			test = True,
			valid_interval = config.valid_interval,
		    log_interval = config.log_interval,
			save_path = config.save_path,
			use_early_stopping = config.use_early_stopping,
			metric = config.metric,
			patience = config.patience,
			delta = config.delta,
			use_wandb = True)
		trainer.run()