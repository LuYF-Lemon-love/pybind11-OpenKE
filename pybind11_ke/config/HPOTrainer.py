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
import pprint
from ..utils import import_class
from ..config import Trainer, Tester
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from ..data import TrainDataLoader, TestDataLoader

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import TransE
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

def get_hpo_config():

	"""返回超参数优化配置的默认优化参数。
	
	:returns: 超参数优化配置的默认优化参数
	:rtype: dict
	"""

	wandb.login()

	sweep_config = {
		'method': 'bayes',
		'name': 'pybind11_ke_hpo'
	}

	metric = {
		'name': 'val/hit10',
		'goal': 'maximize'
	}

	# parameters_dict = {
	# 	'in_path': {
	# 		'value': './benchmarks/FB15K/'
	# 	},
	# 	'ent_file': {
	# 		'value': 'entity2id.txt'
	# 	},
	# 	'rel_file': {
	# 		'value': 'relation2id.txt'
	# 	},
	# 	'train_file': {
	# 		'value': 'train2id.txt'
	# 	},
	# 	'batch_size': {
	# 		'value': None
	# 	},
	# 	'nbatches': {
	# 		'value': 100
	# 	},
	# 	'threads': {
	# 		'value': 8
	# 	},
	# 	'sampling_mode': {
	# 		'value': 'normal'
	# 	},
	# 	'bern': {
	# 		'value': True
	# 	},
	# 	'neg_ent': {
	# 		'value': [1, 10, 20]
	# 	},
	#     'neg_rel': {
	#         'values': 0
	#     },
	# 	'model': {
	#         'values': 'TransE'
	#     },
	# 	# 'dim': {
	#     #     'values': 50
	#     # },
	# 	'p_norm': {
	#         'values': 1
	#     },
	# 	'norm_flag': {
	#         'values': True
	#     },
	# 	'loss': {
	#         'values': 'MarginLoss'
	#     },
	# 	# 'adv_temperature': {
	#     #     'values': None
	#     # },
	# 	'margin': {
	#         'values': 1.0
	#     },
	# 	'regul_rate': {
	#         'values': 0.0
	#     },
	# 	'l3_regul_rate': {
	#         'values': 0.0
	#     },
	# 	'valid_file': {
	# 		'value': 'valid2id.txt'
	# 	},
	# 	'test_file': {
	# 		'value': 'test2id.txt'
	# 	},
	# 	'type_constrain': {
	#         'values': True
	#     },
	# 	'use_gpu': {
	#         'values': True
	#     },
	# 	# 'device': {
	#     #     'values': 'cuda:0'
	#     # },
	# 	'epochs': {
	#         'values': 50
	#     },
	# 	'lr': {
	#         'distribution': 'uniform',
	#         'min': 0,
	#         'max': 1.0
	#     },
	# 	'opt_method': {
	#         'values': ['adam', 'sgd']
	#     },
	#     'test': {
	#         'values': True
	#     },
	# 	'valid_interval': {
	#         'values': 10
	#     },
	# 	'log_interval': {
	#         'values': 10
	#     },
	# 	'save_interval': {
	#         'values': None
	#     },
	# 	'save_path': {
	#         'values': './checkpoint/transe.pth'
	#     },
	# 	'use_wandb': {
	#         'values': True
	#     },
	# }

	parameters_dict = {
		'nbatches': {
			'value': 100
		},
		'threads': {
			'value': 8
		},
		'sampling_mode': {
			'value': 'normal'
		},
		'bern': {
			'value': True
		},
		'p_norm': {
	        'values': [1, 2]
	    },
	}

	sweep_config['metric'] = metric
	sweep_config['parameters'] = parameters_dict

	pprint.pprint(sweep_config)

	sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

	wandb.agent(sweep_id, hpo_train, count=1)

	return sweep_config

def hpo_train(config=None):

	"""超参数优化训练循环函数。
	
	:param config: wandb 的项目配置如超参数。
	:type config: dict
	"""
	
	with wandb.init(config=config):
		
		config = wandb.config

		# dataloader for training
		train_dataloader = TrainDataLoader(
		    in_path = "./benchmarks/FB15K/",
		    nbatches = config.nbatches,
		    threads = config.threads,
		    sampling_mode = config.sampling_mode,
		    bern = config.bern,
		    neg_ent = 25,
		    neg_rel = 0)

		# define the model
		transe = TransE(
		    ent_tot = train_dataloader.get_ent_tol(),
		    rel_tot = train_dataloader.get_rel_tol(),
		    dim = 50,
		    p_norm = config.p_norm,
		    norm_flag = True)

		# define the loss function
		model = NegativeSampling(
		    model = transe,
		    loss = MarginLoss(margin = 1.0),
		    batch_size = train_dataloader.get_batch_size()
		)

		# dataloader for test
		test_dataloader = TestDataLoader('./benchmarks/FB15K/')

		# test the model
		tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True, device = 'cuda:1')

		# train the model
		trainer = Trainer(model = model, data_loader = train_dataloader,
		    epochs = 1000, lr = 0.01, use_gpu = True, device = 'cuda:1',
		    tester = tester, test = True, valid_interval = 100,
		    log_interval = 100, save_interval = 100, save_path = './checkpoint/transe.pth', use_wandb=True)
		trainer.run()

		# # dataloader for training
		# train_dataloader = TrainDataLoader(
		#     in_path = config.in_path,
		# 	ent_file = config.ent_file,
		# 	rel_file = config.rel_file,
		# 	train_file = config.train_file,
		# 	batch_size = config.batch_size,
		# 	nbatches = config.nbatches,
		# 	threads = config.threads,
		# 	sampling_mode = config.sampling_mode,
		# 	bern = config.bern,
		# 	neg_ent = config.neg_ent,
		# 	neg_rel = config.neg_rel)

		# model_class = import_class(f"..module.model.{config.model}")

		# # define the model
		# kge_model = model_class(
		#     ent_tot = train_dataloader.get_ent_tol(),
		#     rel_tot = train_dataloader.get_rel_tol(),
		#     # dim = config.dim,
		#     p_norm = config.p_norm,
		#     norm_flag = config.norm_flag)

		# loss_class = import_class(f"..module.loss.{config.loss}")

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