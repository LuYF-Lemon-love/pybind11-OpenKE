pybind11_ke.module.model
===================================

.. automodule:: pybind11_ke.module.model

.. contents:: pybind11_ke.module.model
    :depth: 3
    :local:
    :backlinks: top

基础模块
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    CompGCNCov

模型基类
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    Model

平移模型
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    TransE
    TransH
    TransR
    TransD
    RotatE

语义匹配模型
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    RESCAL
    DistMult
    HolE
    ComplEx
    Analogy
    SimplE

图神经网络模型
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    RGCN
    CompGCN

超参数优化默认搜索范围
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    get_transe_hpo_config
    get_transh_hpo_config