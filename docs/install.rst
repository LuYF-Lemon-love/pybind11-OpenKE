安装
==================================

Pip
----------------------------------

1. 安装：

.. prompt:: bash

    pip install dgl
    pip install git+https://github.com/LuYF-Lemon-love/pybind11-OpenKE.git

2. 验证:

::

    >>> import pybind11_ke
    >>> pybind11_ke.__version__
    '3.0.0'
    >>>

Linux
----------------------------------

1. 克隆 pybind11-OpenKE-PyTorch 分支。

.. prompt:: bash

    git clone -b main git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    cd pybind11-OpenKE/
    python -m venv env
    source env/bin/activate
    which python
    pip install --upgrade pip
    pip install dgl
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 快速开始。

.. prompt:: bash

    cd examples/TransE/
    python single_gpu_transe_FB15K.py

Windows
----------------------------------

1. 克隆 pybind11-OpenKE-PyTorch 分支。

.. prompt:: bash

    git clone -b main git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    cd pybind11-OpenKE/
    py -m venv env
    .\env\Scripts\activate
    pip install --upgrade pip
    pip install dgl
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 快速开始。

.. prompt:: bash

    cd examples/TransE/
    python single_gpu_transe_FB15K.py