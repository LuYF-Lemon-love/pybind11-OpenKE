安装
==================================

Linux
----------------------------------

1. 克隆 pybind11-OpenKE-PyTorch 分支。

.. prompt:: bash

    git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    cd pybind11-OpenKE/
    python -m venv env
    source env/bin/activate
    which python
    pip install --upgrade pip
    pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 快速开始。

.. prompt:: bash

    cd experiments/TransE/
    python single_gpu_transe_FB15K.py