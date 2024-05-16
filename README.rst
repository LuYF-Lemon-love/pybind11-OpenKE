.. figure:: https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/pybind11-OpenKE@main/docs/_static/images/logo-best.svg
    :alt: pybind11-OpenKE logo

pybind11-OpenKE — 知识图谱嵌入工具包
----------------------------------------------

.. image:: https://readthedocs.org/projects/pybind11-openke/badge/?version=latest
    :target: https://pybind11-openke.readthedocs.io/zh_CN/latest/?badge=latest
    :alt: Documentation Status

基于 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 开发的知识图谱嵌入工具包，支持跨平台运行，具备自动超参数搜索、高效并行训练以及实验结果记录功能，为研究与应用提供强大助力。

教程和 API 参考文档可以访问 
`pybind11-openke.readthedocs.io <https://pybind11-openke.readthedocs.io/zh_CN/latest/>`_。
源代码可以访问 `github.com/LuYF-Lemon-love/pybind11-OpenKE <https://github.com/LuYF-Lemon-love/pybind11-OpenKE>`_。

📁 `pybind11_ke/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/pybind11_ke/>`_
    pybind11-OpenKE 源代码保存在 ``pybind11_ke/``。

📚 `docs/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/docs/>`_
    所有的文档源文件保存在 ``docs/``。 所有的 ``*.rst`` 构成了文档中的各个部分。

🌰 `examples/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/examples/>`_
    pybind11-OpenKE 的例子保存在 ``examples/``，修改自 ``OpenKE-PyTorch``。

📍 `logs/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/logs/>`_
    pybind11-OpenKE 的例子运行日志保存在 ``logs/``。

💡 `benchmarks/ <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/benchmarks/>`_
    常用的知识图谱保存在 ``benchmarks/``。

📜 `README.rst <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/main/README.rst>`_
    项目主页。
    
⁉️ Questions / comments
    如果你有任何问题，可以在 `Github issue <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/issues>`_ 提问。

.. Note:: 本项目基于 OpenKE-PyTorch 的版本保存在 `thunlp-OpenKE-PyTorch <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/thunlp-OpenKE-PyTorch>`_ 分支。

新特性
------------

**易用的**

- **1.0.0 版本**: 利用 C++ 重写底层数据处理、C++11 的线程库实现并行、 `pybind11 <https://github.com/pybind/pybind11>`__ 实现 Python 和 C++ 的交互，进而能够做到跨平台 (Windows, Linux)。

- **2.0.0 版本**: 使用 Python 重写底层数据处理，进而能够做到跨平台 (Windows, Linux)。

- 使用 `Setuptools <https://setuptools.pypa.io/en/latest/>`__ 打包了 pybind11-OpenKE， 使得能够像其他第三方库一样使用。

- 增加了文档。

**正确的**

- 增加了 ``R-GCN`` :cite:`R-GCN` 模型。

- 增加了 ``CompGCN`` :cite:`CompGCN` 模型。

- 修复了 `SimplE模型实现的问题 <https://github.com/thunlp/OpenKE/issues/151>`__ 。

- 修复了 :ref:`HolE <details_hole>` 深度学习框架（pytorch）的版本适配问题。

**高效的**

- 使用 :py:class:`torch.nn.parallel.DistributedDataParallel` 完成数据并行（ **2.0.0 版本** 使用 `accelerate <https://github.com/huggingface/accelerate>`_ 实现），使得 ``pybind11-OpenKE`` 能够利用多个 ``GPU`` 同时训练。

- 增加超参数扫描功能（随机搜索、网格搜索和贝叶斯搜索）。

**扩展的**

- 在模型训练过程中，能够在验证集上评估模型（模型能够一次评估多个三元组（batch），能够大大加速模型评估）。

- 增加了学习率调度器。

- 能够利用 `wandb <https://wandb.ai/>`_ 输出日志。

- 实现了早停止。

- 能够自定义 Hits@N。

支持的知识图谱嵌入模型：

.. list-table::
    :widths: 20 50
    :header-rows: 1

    * - 类型
      - 模型
    * - 平移模型
      - ``TransE`` :cite:`TransE`, ``TransH`` :cite:`TransH`, ``TransR`` :cite:`TransR`, ``TransD`` :cite:`TransD`, ``RotatE`` :cite:`RotatE`
    * - 语义匹配模型
      - ``RESCAL`` :cite:`RESCAL`, ``DistMult`` :cite:`DistMult`, ``HolE`` :cite:`HolE`, ``ComplEx`` :cite:`ComplEx`, ``Analogy`` :cite:`ANALOGY`, ``SimplE`` :cite:`SimplE`
    * - 图神经网络模型
      - ``R-GCN`` :cite:`R-GCN`, ``CompGCN`` :cite:`CompGCN`