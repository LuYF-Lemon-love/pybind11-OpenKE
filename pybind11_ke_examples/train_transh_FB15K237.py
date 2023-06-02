"""
`RESCAL-FB15K237 <train_rescal_FB15K237.html>`_ ||
**TransE-FB15K237** ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

TransE-FB15K237
===================
这一部分介绍如何用在 FB15K237 知识图谱上训练 TransE。

TransE 原论文: `Translating Embeddings for Modeling Multi-relational Data <https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`__。

导入数据
-----------------
pybind11-OpenKE 有两个工具用于导入数据: :py:class:`pybind11_ke.data.TrainDataLoader` 和
:py:class:`pybind11_ke.data.TestDataLoader`。

"""

from pybind11_ke.config import Trainer, Tester
from pybind11_ke.module.model import TransH
from pybind11_ke.module.loss import MarginLoss
from pybind11_ke.module.strategy import NegativeSampling
from pybind11_ke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("../benchmarks/FB15K237/", "link")

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transh.save_checkpoint('../checkpoint/transh.ckpt')

# test the model
transh.load_checkpoint('../checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)