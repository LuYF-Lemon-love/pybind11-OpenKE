数据集
==================================

包含的论文数据集
----------------------------------------

.. list-table::
    :widths: 10 10 10 10 10 10 10
    :header-rows: 1

    * - 名称
      - 实体
      - 关系
      - 训练集
      - 验证集
      - 测试集
      - 原论文
    * - `WN18 <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/WN18>`_
      - 40,943
      - 18
      - 141,442
      - 5,000
      - 5,000
      - ``TransE`` :cite:`TransE`
    * - `WN18RR <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/WN18RR>`_
      - 40,943
      - 11
      - 86,835
      - 3,034
      - 3,134
      - 未知
    * - `FB15K <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/FB15K>`_
      - 14,951
      - 1,345
      - 483,142
      - 50,000
      - 59,071
      - ``TransE`` :cite:`TransE`
    * - `FB15k-237 <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/benchmarks/FB15K237>`_
      - 14,541
      - 237
      - 272,115
      - 17,535
      - 20,466
      - 未知

数据集格式
----------------------------------------

* 对于训练模型，数据集包含 3 个文件：

  - train2id.txt：训练集文件，第一行是训练集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组，表示在实体 **e1** 和实体 **e2** 之间有一个关系 **rel**。

  - entity2id.txt：第一行是实体的个数。其余行是全部实体和相应的 id，每一行一个实体。

  - relation2id.txt：第一行是关系的个数。其余行是全部关系和相应的 id，每一行一个关系。

* 对于验证模型，需要 2 个额外的文件（总共 5 个文件）。

  - valid2id.txt：验证集文件，第一行是验证集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组。

  - test2id.txt：测试集文件，第一行是测试集中三元组的个数。后面所有行都是 **(e1, e2, rel)** 格式的三元组。

  - type_constrain.txt: 类型约束文件，第一行是关系的个数。后面所有行是每个关系的类型约束。如 benchmarks/FB15K 的 id 为 1200 的关系，它有 4 种类型头实体（3123，1034，58 和 5733）和 4 种类型的尾实体（12123，4388，11087 和 11088）。

.. Note:: train2id.txt、valid2id.txt 和 test2id.txt 包含的是来自 entitiy2id.txt 和 relation2id.txt 的 id，
    而不是实体和关系的名字。

.. Note:: type_constrain.txt 可以通过 benchmarks/FB15K/n-n.py 脚本获得。