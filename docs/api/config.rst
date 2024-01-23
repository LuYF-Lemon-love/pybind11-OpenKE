pybind11_ke.config
===================================

.. automodule:: pybind11_ke.config

.. contents:: pybind11_ke.config
    :depth: 2
    :local:
    :backlinks: top

训练循环
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Trainer
    GraphTrainer

评估循环
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Tester
    GraphTester

链接预测函数
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    link_predict
    head_predict
    tail_predict
    calc_ranks

并行训练函数
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    trainer_distributed_data_parallel

超参数优化默认搜索范围
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst
    
    get_tester_hpo_config
    get_trainer_hpo_config
    get_graph_tester_hpo_config
    get_graph_trainer_hpo_config

超参数优化训练循环函数
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    set_hpo_config
    start_hpo_train
    hpo_train