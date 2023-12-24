.. figure:: https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/pybind11-OpenKE@pybind11-OpenKE-PyTorch/docs/_static/logo-best.svg
    :alt: pybind11-OpenKE logo

pybind11-OpenKE — 知识图谱嵌入学习包
----------------------------------------------

.. image:: https://readthedocs.org/projects/pybind11-openke/badge/?version=latest
    :target: https://pybind11-openke.readthedocs.io/zh_CN/latest/?badge=latest
    :alt: Documentation Status

基于 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 开发的知识图谱嵌入学习包，
底层数据处理利用 C++ 实现，使用 `pybind11 <https://github.com/pybind/pybind11>`__ 实现 C++ 和 Python 的交互。

不久后将完成, 稍事等待.

教程和 API 参考文档可以访问 
`pybind11-openke.readthedocs.io <https://pybind11-openke.readthedocs.io/zh_CN/latest/>`_。
源代码可以访问 `github.com/LuYF-Lemon-love/pybind11-OpenKE <https://github.com/LuYF-Lemon-love/pybind11-OpenKE>`_。

📁 `pybind11_ke/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/pybind11_ke/>`_
    pybind11-OpenKE 源代码保存在 ``pybind11_ke/``。

📚 `docs/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/docs/>`_
    所有的文档源文件保存在 ``docs/``。 所有的 ``*.rst`` 构成了文档中的各个部分。

🌰 `examples/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/examples/>`_
    pybind11-OpenKE 的例子保存在 ``examples/``，取自 ``OpenKE-PyTorch``。

💡 `benchmarks <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/>`_
    常用的知识图谱保存在 ``benchmarks/``。

🍋 `result <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/result>`_
    OpenKE-PyTorch 和 pybind11-OpenKE 在我们机器上运行的结果保存在 ``result/``。

📍 `environment/requirements.txt <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/environment/requirements.txt>`_ 
    在我们机器上的 Python 的依赖，可以作为你的参考，保存在 ``requirements.txt``。

📜 `README.rst <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/README.rst>`_
    项目主页。
    
⁉️ Questions / comments
    如果你有任何问题，可以在 `Github issue <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/issues>`_ 提问。

.. Note:: 本项目基于 OpenKE-PyTorch 的版本保存在 `thunlp-OpenKE-PyTorch <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/thunlp-OpenKE-PyTorch>`_ 分支。

New Features
------------

**易用的**

- 利用 C++ 重写底层数据处理、C++11 的线程库实现并行、pybind11 实现 Python 和 C++ 的交互，进而能够做到跨平台 (Windows, Linux)。

- 使用 `Setuptools <https://setuptools.pypa.io/en/latest/>`__ 打包了 pybind11-OpenKE， 使得能够像其他第三方库一样使用。

- 增加了文档。

**正确的**

- 修复 `SimplE模型实现的问题 <https://github.com/thunlp/OpenKE/issues/151>`__ 。

**高效的**

- 使用 :py:class:`torch.nn.parallel.DistributedDataParallel` 完成数据并行，使得 ``pybind11-OpenKE`` 能够利用多个 ``GPU`` 同时训练。

**扩展的**

- 在模型训练过程中，能够在验证集上评估模型。

OpenKE-PyTorch
--------------

OpenKE-PyTorch 是一个基于 PyTorch 实现的知识图谱嵌入的开源框架。

支持的知识图谱嵌入模型：

- RESCAL: `A Three-Way Model for Collective Learning on Multi-Relational Data <https://icml.cc/Conferences/2011/papers/438_icmlpaper.pdf>`__ .

- TransE: `Translating Embeddings for Modeling Multi-relational Data <https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`__ .

- TransH: `Knowledge Graph Embedding by Translating on Hyperplanes <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`__ .

- DistMult: `Embedding Entities and Relations for Learning and Inference in Knowledge Bases <https://arxiv.org/abs/1412.6575>`__ .

- TransR: `Learning Entity and Relation Embeddings for Knowledge Graph Completion <https://ojs.aaai.org/index.php/AAAI/article/view/9491>`__ .

- TransD: `Knowledge Graph Embedding via Dynamic Mapping Matrix <https://aclanthology.org/P15-1067/>`__ .

- HolE: `Holographic Embeddings of Knowledge Graphs <https://ojs.aaai.org/index.php/AAAI/article/view/10314>`__ .

- ComplEx: `Complex Embeddings for Simple Link Prediction <https://arxiv.org/abs/1606.06357>`__ .

- ANALOGY: `Analogical Inference for Multi-relational Embeddings <https://proceedings.mlr.press/v70/liu17d.html>`__ .

- SimplE: `SimplE Embedding for Link Prediction in Knowledge Graphs <https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html>`__ .

- RotatE: `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space <https://openreview.net/forum?id=HkgEQnRqYQ>`__ .

实验设置
---------------------

对于每一个测试三元组，头实体依次被实体集中的每一个实体替换。
这些损坏的三元组首先被 KGE 模型计算得分，然后按照得分排序。
我们因此得到了正确实体的排名。上述过程通过替换尾实体被重复。
最后，报道了在上述设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10, Hits@3, Hits@1），
平均排名（mean rank，MR），平均倒数排名（mean reciprocal rank，MRR）。

因为一些损坏的三元组可能已经存在训练集和验证集中，在这种情形下，
这些损坏三元组可能比测试集中的三元组排名更靠前，但不应该被认为是
错误的，因为两个三元组都是正确的。我们删除了那些出现在训练集、验证集或测试集中的损坏的三元组，
这确保了这样的损坏三元组不会参与排名。最后，报告了在这种设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10 (filter), Hits@3(filter), Hits@1(filter)），
平均排名（mean rank，MR(filter)），平均倒数排名（mean reciprocal rank，MRR(filter)）。

上述设置更多的细节可以从 `TransE 的原论文 <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`__ 获得。

对于大型知识图谱，用整个实体集合损坏三元组是极其耗时的。
因此 ``OpenKE-PyTorch`` 提供了名为
"`type constraint <https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf>`__"
的实验性的设置用有限的实体集合（取决于相应的关系）损坏三元组。

安装 (Linux)
--------------------

.. WARNING:: 由于 :py:class:`pybind11_ke.module.model.HolE` 的
    :py:meth:`pybind11_ke.module.model.HolE._ccorr` 需要
    :py:func:`torch.rfft` 和 :py:func:`torch.ifft` 分别计算实数到复数离散傅里叶变换和复数到复数离散傅立叶逆变换。
    ``pytorch`` 在版本 ``1.8.0`` 移除了上述两个函数，并且在版本 ``1.7.0`` 给出了警告。
    因此，我们强烈建议安装版本 ``1.6.0``。我们将不久以后修改
    :py:class:`pybind11_ke.module.model.HolE`，
    使得能够适配更高版本的 ``pytorch``。

1. 配置环境：

.. code-block:: console

    $ conda create --name pybind11-ke python=3.8 -y
    $ conda activate pybind11-ke
    $ pip install torch==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 克隆 pybind11-OpenKE-PyTorch 分支。

.. code-block:: console

    $ git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    $ cd pybind11-OpenKE/
    $ pip install .

3. 快速开始。

.. code-block:: console

    $ cd experiments/TransE/
    $ python single_gpu_transe_FB15K.py

数据
----

* 对于训练模型，数据集包含 3 个文件：

  - ``train2id.txt``：训练集文件，第一行是训练集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组，表示在实体 **e1** 和实体 **e2** 之间有一个关系 **rel**。

  - ``entity2id.txt``：第一行是实体的个数。其余行是全部实体和相应的 id，每一行一个实体。

  - ``relation2id.txt``：第一行是关系的个数。其余行是全部关系和相应的 id，每一行一个关系。

* 对于验证模型，需要 2 个额外的文件（总共 5 个文件）。

  - ``test2id.txt``：测试集文件，第一行是测试集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组。

  - ``valid2id.txt``：验证集文件，第一行是验证集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组。

  - ``type_constrain.txt``: 类型约束文件，第一行是关系的个数。后面所有行是每个关系的类型约束。如 ``benchmarks/FB15K`` 的 id 为 1200 的关系，它有 4 种类型头实体（3123，1034，58 和 5733）和 4 种类型的尾实体（12123，4388，11087 和 11088）。

.. Note:: ``train2id.txt`` 包含的是来自 ``entitiy2id.txt`` 和 ``relation2id.txt`` 的 id，
    而不是实体和关系的名字。

.. Note:: ``type_constrain.txt`` 可以通过 ``benchmarks/FB15K/n-n.py`` 脚本获得。

参考
---------

#. `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__.

#. `pybind11 <https://github.com/pybind/pybind11>`__.

#. `Setuptools <https://setuptools.pypa.io/en/latest/>`__.