pybind11_ke.data
===================================

.. automodule:: pybind11_ke.data

.. contents:: pybind11_ke.data
    :depth: 2
    :local:
    :backlinks: top

平移模型和语义匹配模型数据加载器
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TrainDataSampler
    TrainDataLoader
    TestDataSampler
    TestDataLoader

图神经网络模型数据加载器
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    KGReader
    RevSampler
    GraphSampler
    CompGCNSampler
    GraphTestSampler
    CompGCNTestSampler
    GraphDataLoader

超参数优化默认搜索范围
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    get_train_data_loader_hpo_config
    get_test_data_loader_hpo_config
    get_graph_data_loader_hpo_config