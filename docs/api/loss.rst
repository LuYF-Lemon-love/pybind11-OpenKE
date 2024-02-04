pybind11_ke.module.loss
===================================

.. automodule:: pybind11_ke.module.loss

.. contents:: pybind11_ke.module.loss
    :depth: 3
    :local:
    :backlinks: top

损失函数基类
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    Loss

损失函数子类
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    MarginLoss
    SigmoidLoss
    SoftplusLoss
    RGCNLoss
    CompGCNLoss

超参数优化默认搜索范围
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    get_margin_loss_hpo_config
    get_sigmoid_loss_hpo_config
    get_softplus_loss_hpo_config
    get_rgcn_loss_hpo_config
    get_cross_entropy_loss_hpo_config