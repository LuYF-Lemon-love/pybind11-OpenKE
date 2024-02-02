# coding:utf-8
#
# pybind11_ke/config/__init__.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 3, 2023
# 
# 该头文件定义了 config 接口.

"""循环部分，包含训练循环和验证循环。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Trainer import Trainer, get_trainer_hpo_config
from .Tester import Tester, get_tester_hpo_config, link_predict, head_predict, tail_predict, calc_ranks

# from .TrainerDataParallel import trainer_distributed_data_parallel
from .HPOTrainer import set_hpo_config, set_hpo_hits,start_hpo_train, hpo_train

__all__ = [
	'Trainer',
	'get_trainer_hpo_config',
	'Tester',
	'get_tester_hpo_config',
	'link_predict',
	'head_predict',
	'tail_predict',
	'calc_ranks',
	# 'trainer_distributed_data_parallel',
	'set_hpo_config',
	'set_hpo_hits',
	'start_hpo_train',
	'hpo_train'
]
