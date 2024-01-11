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
   :caption: TransE

   experiments/TransE/single_gpu_transe_FB15K
   experiments/TransE/single_gpu_transe_FB15K_wandb
   experiments/TransE/single_gpu_transe_FB15K_hpo
   experiments/TransE/multigpu_transe_FB15K
   experiments/TransE/multigpu_transe_FB15K_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: TransH

   experiments/TransH/single_gpu_transh_FB15K237
   experiments/TransH/single_gpu_transh_FB15K237_wandb
   experiments/TransH/multigpu_transh_FB15K237
   experiments/TransH/multigpu_transh_FB15K237_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: TransR

   experiments/TransR/single_gpu_transr_FB15K237
   experiments/TransR/multigpu_transr_FB15K237

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: TransD

   experiments/TransD/single_gpu_transd_FB15K237_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: RotatE

   experiments/RotatE/single_gpu_rotate_WN18RR_adv
   experiments/RotatE/single_gpu_rotate_WN18RR_adv_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: DistMult

   experiments/DistMult/single_gpu_distmult_WN18RR_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: HolE

   experiments/HolE/single_gpu_hole_WN18RR_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: ComplEx

   experiments/ComplEx/single_gpu_complex_WN18RR_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: SimplE

   experiments/SimplE/single_gpu_simple_WN18RR_wandb

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: OpenKE 的例子

   examples/train_rescal_FB15K237
   examples/train_transe_FB15K237
   examples/train_transe_WN18_adv_sigmoidloss
   examples/train_transh_FB15K237
   examples/train_distmult_WN18RR
   examples/train_distmult_WN18RR_adv
   examples/train_transd_FB15K237
   examples/train_hole_WN18RR
   examples/train_complex_WN18RR
   examples/train_analogy_WN18RR
   examples/train_simple_WN18RR
   examples/train_rotate_WN18RR_adv

.. toctree::
   :maxdepth: 4
   :caption: API

   api

.. toctree::
   :maxdepth: 1
   :caption: 参考

   reference