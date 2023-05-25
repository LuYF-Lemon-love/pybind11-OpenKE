Usage
=====

.. _安装:

安装
----

1. 配置环境：

.. code-block:: console

    $ conda create --name pybind11-ke python=3.10 -y
    $ conda activate pybind11-ke
    $ pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
    $ pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 克隆 pybind11-OpenKE-PyTorch 分支。

.. code-block:: console

    $ git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    $ cd pybind11-OpenKE/
    $ mkdir -p ./checkpoint
    $ cd pybind11_ke/

3. 编译 C++ 文件。

.. code-block:: console

    $ bash make.sh

4. 快速开始。

.. code-block:: console

    $ cd ../
    $ cp examples/train_transe_FB15K237.py ./
    $ python train_transe_FB15K237.py

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

Trainer
-------

.. automodule:: Trainer
   :members:

.. autoclass:: Trainer
   :members: