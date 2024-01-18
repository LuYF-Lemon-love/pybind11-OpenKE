.. figure:: https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/pybind11-OpenKE@pybind11-OpenKE-PyTorch/docs/_static/logo-best.svg
    :alt: pybind11-OpenKE logo

pybind11-OpenKE — 知识图谱嵌入工具包
----------------------------------------------

.. image:: https://readthedocs.org/projects/pybind11-openke/badge/?version=latest
    :target: https://pybind11-openke.readthedocs.io/zh_CN/latest/?badge=latest
    :alt: Documentation Status

基于 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 开发的知识图谱嵌入工具包，
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
    pybind11-OpenKE 的例子保存在 ``examples/``，修改自 ``OpenKE-PyTorch``。

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

新特性
------------

**易用的**

- 利用 C++ 重写底层数据处理、C++11 的线程库实现并行、pybind11 实现 Python 和 C++ 的交互，进而能够做到跨平台 (Windows, Linux)。

- 使用 `Setuptools <https://setuptools.pypa.io/en/latest/>`__ 打包了 pybind11-OpenKE， 使得能够像其他第三方库一样使用。

- 增加了文档。

**正确的**

- 增加了 ``R-GCN`` :cite:`R-GCN` 模型。

- 修复了 `SimplE模型实现的问题 <https://github.com/thunlp/OpenKE/issues/151>`__ 。

- 修复了 :ref:`HolE <details_hole>` 深度学习框架（pytorch）的版本适配问题。

**高效的**

- 使用 :py:class:`torch.nn.parallel.DistributedDataParallel` 完成数据并行，使得 ``pybind11-OpenKE`` 能够利用多个 ``GPU`` 同时训练。

- 增加超参数扫描功能（随机搜索、网格搜索和贝叶斯搜索）。

**扩展的**

- 在模型训练过程中，能够在验证集上评估模型。

- 增加了学习率调度器。

- 能够利用 `wandb <https://wandb.ai/>`_ 输出日志。

- 实现了早停止。

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
      - ``R-GCN`` :cite:`R-GCN`