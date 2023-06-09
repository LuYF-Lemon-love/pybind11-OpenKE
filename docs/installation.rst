安装
=====

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
    $ mkdir -p ./checkpoint
    $ pip install .

3. 快速开始。

.. code-block:: console

    $ cd pybind11_ke_examples/
    $ python train_transe_FB15K237.py
