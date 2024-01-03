# coding:utf-8
#
# pybind11_ke/config/HPOTrainer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 2, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 2, 2023
#
# 该脚本定义了并行训练循环函数.

"""
hpo_train - 超参数优化训练循环函数。
"""

import wandb
from ..utils import import_class
from ..config import Trainer, Tester
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from ..data import TrainDataLoader, TestDataLoader

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TestDataLoader

def set_hpo_config(
	method = 'bayes',
	sweep_name = 'pybind11_ke_hpo',
	metric_name = 'val/hit10',
	metric_goal = 'maximize',
	train_data_loader_config = None,
	kge_config = None,
	loss_config = None):

	"""返回超参数优化配置的默认优化参数。
	
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
	:returns: 超参数优化配置的默认优化参数
	:rtype: dict
	"""

	sweep_config = {
		'method': method,
		'name': sweep_name
	}

	metric = {
		'name': metric_name,
		'goal': metric_goal
	}

	parameters_dict = {}
	parameters_dict.update(train_data_loader_config)
	parameters_dict.update(kge_config)
	parameters_dict.update(loss_config)

	sweep_config['metric'] = metric
	sweep_config['parameters'] = parameters_dict

	return sweep_config

def start_hpo_train(config = None, project = "pybind11-ke-sweeps", count = 2):

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

def hpo_train(config=None):

	"""超参数优化训练循环函数。
	
	:param config: wandb 的项目配置如超参数。
	:type config: dict
	"""
	
	with wandb.init(config=config):
		
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
		kge_model = model_class(
		    ent_tot = train_dataloader.get_ent_tol(),
		    rel_tot = train_dataloader.get_rel_tol(),
		    dim = config.dim,
		    p_norm = config.p_norm,
		    norm_flag = config.norm_flag)

		# define the loss function
		loss_class = import_class(f"pybind11_ke.module.loss.{config.loss}")
		model = NegativeSampling(
		    model = kge_model,
		    loss = loss_class(
				adv_temperature = config.adv_temperature,
				margin = config.margin
				),
		    batch_size = train_dataloader.get_batch_size()
		)

		# dataloader for test
		test_dataloader = TestDataLoader('./benchmarks/FB15K/')

		# test the model
		tester = Tester(model = kge_model, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
		    epochs = 50, lr = 0.01, use_gpu = True, device = 'cuda:1',
		    tester = tester, test = True, valid_interval = 10,
		    log_interval = 10, save_interval = 10, save_path = './checkpoint/transe.pth', use_wandb=True)
		trainer.run()

		# # define the loss function
		# model = NegativeSampling(
		#     model = kge_model,
		#     loss = loss_class(
		# 		# adv_temperature = config.adv_temperature,
		# 		margin = config.margin),
		#     batch_size = train_dataloader.get_batch_size(),
		# 	regul_rate = config.regul_rate,
		# 	l3_regul_rate = config.l3_regul_rate
		# )

		# # dataloader for test
		# test_dataloader = TestDataLoader(
		# 	in_path = train_dataloader.in_path,
		# 	ent_file = train_dataloader.ent_file,
		# 	rel_file = train_dataloader.rel_file,
		# 	train_file = train_dataloader.train_file,
		# 	valid_file = config.valid_file,
		# 	test_file = config.test_file,
		# 	type_constrain = config.type_constrain
		# )

		# # test the model
		# tester = Tester(
		# 	model = kge_model,
		# 	data_loader = test_dataloader,
		# 	use_gpu = config.use_gpu,
		# 	# device = config.device
		# )

		# # train the model
		# trainer = Trainer(
		# 	model = model,
		# 	data_loader = train_dataloader,
		#     epochs = config.epochs,
		# 	lr = config.lr,
		# 	opt_method = config.opt_method,
		# 	use_gpu = config.use_gpu,
		# 	# device = config.device,
		#     tester = tester,
		# 	test = config.test,
		# 	valid_interval = config.valid_interval,
		#     log_interval = config.log_interval,
		# 	save_interval = config.save_interval,
		#     save_path = config.save_path,
		# 	use_wandb = True)
		# trainer.run()