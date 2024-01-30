pybind11_ke.data
======================================================================

.. automodule:: pybind11_ke.data

.. contents:: pybind11_ke.data
    :depth: 2
    :local:
    :backlinks: top

数据预加载器
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    KGReader

平移模型和语义匹配模型训练集数据采样器
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TradSampler
    UniSampler
    BernSampler

图神经网络模型数据采样器
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    RevSampler
    RGCNSampler
    CompGCNSampler

测试集数据采样器
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TestSampler
    TradTestSampler
    RGCNTestSampler
    CompGCNTestSampler

数据加载器
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    KGEDataLoader

超参数优化默认搜索范围
--------------------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    get_kge_data_loader_hpo_config