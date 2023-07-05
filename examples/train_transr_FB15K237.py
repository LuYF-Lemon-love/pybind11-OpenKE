"""
`RESCAL-FB15K237 <train_rescal_FB15K237.html>`_ ||
`TransE-FB15K237 <train_transe_FB15K237.html>`_ ||
`TransE-WN18RR-adv <train_transe_WN18_adv_sigmoidloss.html>`_ ||
`TransH-FB15K237 <train_transh_FB15K237.html>`_ ||
`DistMult-WN18RR <train_distmult_WN18RR.html>`_ ||
`DistMult-WN18RR-adv <train_distmult_WN18RR_adv.html>`_ ||
`TransD-FB15K237 <train_transd_FB15K237.html>`_ ||
`HolE-WN18RR <train_hole_WN18RR.html>`_ ||
`ComplEx-WN18RR <train_complex_WN18RR.html>`_ ||
`Analogy-WN18RR <train_analogy_WN18RR.html>`_ ||
`SimplE-WN18RR <train_simple_WN18RR.html>`_ ||
**RotatE-WN18RR**


RotatE-WN18RR
===================
这一部分介绍如何用在 WN18RR 知识图谱上训练 RotatE。

RotatE 原论文: `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space <https://openreview.net/forum?id=HkgEQnRqYQ>`__ 。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。

"""

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/FB15K237/",
	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200,
	dim_r = 200,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = True)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("./result/transr_transe.json")

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transr.save_checkpoint('./checkpoint/transr.ckpt')

# test the model
transr.load_checkpoint('./checkpoint/transr.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)