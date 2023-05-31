.. figure:: https://github.com/LuYF-Lemon-love/pybind11-OpenKE/raw/pybind11-OpenKE-PyTorch/docs/_static/logo-best.png
   :alt: pybind11-OpenKE logo

**pybind11-OpenKE — 知识图谱嵌入学习包**

.. image:: https://readthedocs.org/projects/pybind11-openke/badge/?version=latest
    :target: https://pybind11-openke.readthedocs.io/zh_CN/latest/?badge=latest
    :alt: Documentation Status

基于 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 开发的知识表示学习包，
底层数据处理利用 C++ 实现，使用 `pybind11 <https://github.com/pybind/pybind11>`__ 实现 C++ 和 Python 的交互。

不久后将完成, 稍事等待.

教程和 API 参考文档可以访问 
`pybind11-openke.readthedocs.io <https://pybind11-openke.readthedocs.io/zh_CN/latest/>`_。
源代码可以访问 `github.com/LuYF-Lemon-love/pybind11-OpenKE <https://github.com/LuYF-Lemon-love/pybind11-OpenKE>`_。

📁 `pybind11_ke/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/pybind11_ke/>`_
    pybind11-OpenKE 源代码保存在 ``pybind11_ke/``。

📚 `docs/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/docs/>`_
    所有的文档源文件保存在 ``docs/``。 所有的 ``*.rst`` 构成了文档中的各个部分。

🌰 `pybind11_ke_examples/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/pybind11_ke_examples/>`_
    pybind11-OpenKE 的例子保存在 ``pybind11_ke_examples/``。

💡 `benchmarks <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/>`_
    常用的知识图谱保存在 ``benchmarks/``。

🍋 `result <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/result>`_
    OpenKE-PyTorch 和 pybind11-OpenKE 在我们机器上运行的结果保存在 ``result/``。

📍 `requirements.txt <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/requirements.txt>`_ 
    在我们机器上的 Python 的依赖，可以作为你的参考，保存在 ``requirements.txt``。

📜 `README.rst <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/README.rst>`_
    项目主页。
    
⁉️ Questions / comments
    如果你有任何问题，可以在 `Github issue <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/issues>`_ 提问。

New Features
------------

- 利用 C++ 重写底层数据处理，利用 C++11 的线程库实现并行，进而能够做到跨平台 (Windows, Linux).

- 利用 pybind11 实现 Python 和 C++ 的交互.

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

Experimental Settings
---------------------

对于每一个测试三元组，头实体依次被实体集中的每一个实体替换。
这些损坏的三元组首先被 KGE 模型计算得分，然后按照得分排序。
我们因此得到了正确实体的排名。上述过程通过替换尾实体被重复。
最后，报道了在上述设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10, Hits@3, Hits@1），
平均排名（mean rank，MR），平均倒数排名（mean reciprocal rank，MRR）。

因为一些损坏的三元组可能已经存在训练集和验证集中，在这种情形下，
这些损坏三元组可能比测试集中的三元组排名更靠前，但不应该被认为是
错误的，因为两个三元组都是正确的。我们删除了那些出现在训练集、验证集或测试集中的损坏的三元组，
这确保了这样的损坏三元组不会参与排名。最后，报告了在这种设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10, Hits@3, Hits@1），
平均排名（mean rank，MR），平均倒数排名（mean reciprocal rank，MRR）。

上述设置更多的细节可以从 `TransE <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`__ 获得。

对于大型知识图谱，用整个实体集合损坏三元组是极其耗时的。
因此 ``OpenKE-PyTorch`` 提供了名为
"`type constraint <https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf>`__"
的实验性的设置用有限的实体集合（取决于相应的关系）损坏三元组。

For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. 
The scores of those corrupted triplets are first computed by the models and then sorted by the order. 
Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. 
We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). 
The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. 
In this case, those corrupted triplets may be ranked above the test triplet, 
but this should not be counted as an error because both triplets are true. 
Hence, we remove those corrupted triplets appearing in the training, validation or test set, 
which ensures the corrupted triplets are not in the dataset. 
We report the proportion of those correct entities ranked 
in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. 
The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can 
be found from the paper `TransE <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`__.

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. 
Hence, we also provide the experimental setting 
named "`type constraint <https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf>`__" to 
corrupt entities with some limited entity sets determining by their relations.

Installation (Linux)
--------------------

1. 配置环境：

.. code-block:: console

    $ conda create --name pybind11-ke python=3.10 -y
    $ conda activate pybind11-ke
    $ pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 克隆 pybind11-OpenKE-PyTorch 分支。

.. code-block:: console

    $ git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    $ cd pybind11-OpenKE/
    $ mkdir -p ./checkpoint
    $ pip install .

3. 快速开始。

.. code-block:: console

    $ cd pybind11_ke_examples/
    $ python train_transe_FB15K237.py

Data
----

* 对于训练，数据集包含 3 个文件：

  - ``train2id.txt``：训练集文件，第一行是训练集中三元组的个数。

* For training, datasets contain three files:

  - train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format **(e1, e2, rel)** which indicates there is a relation **rel** between **e1** and **e2** . **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  - entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  - relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  - test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format **(e1, e2, rel)** .

  - valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format **(e1, e2, rel)** .

  - type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K.

Reference
---------

#. `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__.

#. `pybind11 <https://github.com/pybind/pybind11>`__.