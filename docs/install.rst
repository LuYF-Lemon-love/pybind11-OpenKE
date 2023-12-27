安装
==================================

Linux
----------------------------------

.. WARNING:: 由于 :py:class:`pybind11_ke.module.model.HolE` 的
    :py:meth:`pybind11_ke.module.model.HolE._ccorr` 需要
    :py:func:`torch.rfft` 和 :py:func:`torch.ifft` 分别计算实数到复数离散傅里叶变换和复数到复数离散傅立叶逆变换。
    ``pytorch`` 在版本 ``1.8.0`` 移除了上述两个函数，并且在版本 ``1.7.0`` 给出了警告。
    因此，我们强烈建议安装版本 ``1.6.0``。我们将不久以后修改
    :py:class:`pybind11_ke.module.model.HolE`，
    使得能够适配更高版本的 ``pytorch``。

1. 配置环境：

.. prompt:: bash

    conda create --name pybind11-ke python=3.8 -y
    conda activate pybind11-ke
    pip install torch==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 克隆 pybind11-OpenKE-PyTorch 分支。

.. prompt:: bash

    git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
    cd pybind11-OpenKE/
    pip install .

3. 快速开始。

.. prompt:: bash

    cd experiments/TransE/
    python single_gpu_transe_FB15K.py