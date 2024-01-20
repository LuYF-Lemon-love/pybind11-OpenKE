.. include:: ../README.rst

欢迎来到 pybind11-OpenKE 文档！
===================================

**pybind11-OpenKE** 是一个知识图谱嵌入工具包，使用 pybind11 来完成 C++ 数据预处理模块与 Python 用户接口的交互，
使用 C++11 线程标准库进行并行化，能够运行在 Windows 和 Linux 操作系统上。

为了使用 pybind11-OpenKE，请先 :doc:`install` 这个项目。 

.. note::

   这个项目依旧出于积极开发中。

目录
--------

.. toctree::
   :maxdepth: 1
   :caption: 概述

   主页 <self>
   install
   datasets
   metric
   details

.. toctree::
   :maxdepth: 1
   :caption: 教程

   tutorials/models

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 平移模型

   examples/TransE/single_gpu_transe_FB15K
   examples/TransE/single_gpu_transe_FB15K_wandb
   examples/TransE/single_gpu_transe_FB15K_hpo
   examples/TransE/multigpu_transe_FB15K
   examples/TransE/multigpu_transe_FB15K_wandb
   examples/TransE/single_gpu_transe_FB15K237_wandb
   examples/TransE/single_gpu_transe_WN18_adv_sigmoidloss_wandb
   examples/TransH/single_gpu_transh_FB15K237
   examples/TransH/single_gpu_transh_FB15K237_wandb
   examples/TransH/multigpu_transh_FB15K237
   examples/TransH/multigpu_transh_FB15K237_wandb
   examples/TransR/single_gpu_transr_FB15K237
   examples/TransR/multigpu_transr_FB15K237
   examples/TransD/single_gpu_transd_FB15K237_wandb
   examples/RotatE/single_gpu_rotate_WN18RR_adv
   examples/RotatE/single_gpu_rotate_WN18RR_adv_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 语义匹配模型

   examples/RESCAL/single_gpu_rescal_FB15K237
   examples/DistMult/single_gpu_distmult_WN18RR_wandb
   examples/DistMult/single_gpu_distmult_WN18RR_adv_sigmoidloss_wandb
   examples/HolE/single_gpu_hole_WN18RR_wandb
   examples/ComplEx/single_gpu_complex_WN18RR_wandb
   examples/ANALOGY/single_gpu_analogy_WN18RR
   examples/SimplE/single_gpu_simple_WN18RR_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 图神经网络模型

   examples/RGCN/single_gpu_rgcn_FB15K237
   examples/CompGCN/single_gpu_compgcn_FB15K237

.. toctree::
   :maxdepth: 4
   :caption: API

   api

.. toctree::
   :maxdepth: 1
   :caption: 参考

   reference