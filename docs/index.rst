.. include:: ../README.rst

欢迎来到 pybind11-OpenKE 文档！
===================================

**pybind11-OpenKE** 是一个知识图谱嵌入学习包，使用 pybind11 来完成 C++ 数据预处理模块与 Python 用户接口的交互，
使用 C++11 线程标准库进行并行化，能够运行在 Windows 和 Linux 操作系统上。

为了使用 pybind11-OpenKE，请先 :doc:`installation` 这个项目。 

.. note::

   这个项目依旧出于积极开发中。

Contents
--------

.. toctree::

   主页 <self>
   installation
   opinion
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: OpenKE 的例子

   auto_examples/train_rescal_FB15K237
   auto_examples/train_transe_FB15K237
   auto_examples/train_transe_WN18_adv_sigmoidloss
   auto_examples/train_transh_FB15K237
   auto_examples/train_distmult_WN18RR
   auto_examples/train_distmult_WN18RR_adv
   auto_examples/train_transd_FB15K237
   auto_examples/train_hole_WN18RR
   auto_examples/train_complex_WN18RR
   auto_examples/train_analogy_WN18RR
   auto_examples/train_simple_WN18RR
   auto_examples/train_rotate_WN18RR_adv
