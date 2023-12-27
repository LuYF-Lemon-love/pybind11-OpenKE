见解
==================================

Analogy
---------

我去掉了原始 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 的 ``Analogy`` 的
`_calc <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
负号，原因如下：

在旧版的 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)>`__ 中，
`DistMult <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/DistMult.py#L23>`__、
`ComplEx <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/ComplEx.py#L36>`__、
`Analogy <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/Analogy.py#L30>`__ 3 者的
``_calc`` 函数都带了负号，并且在
`Analogy 原论文的实现 <https://github.com/quark0/ANALOGY>`__ 中，
`DistMult <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L487>`__、
`ComplEx <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L527>`__、
`Analogy <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L583>`__ 3 者的
``score`` 函数都未带符号。从原论文中也能发现，三者的评分函数的符号应该是一致的。
但是在新版的 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 中，
三者 `DistMult <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/DistMult.py#L40>`__、
`ComplEx <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/ComplEx.py#L21>`__、
`Analogy <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
`_calc` 函数实现中，仅仅 ``Analogy`` 带了负号。

因此，我最终决定去掉 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 的 ``Analogy`` 的
`_calc <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
负号。

从 `运行结果 <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/tree/pybind11-OpenKE-PyTorch/result>`_ 也没发现差异。 

最终实现可以从 `这里 <_modules/pybind11_ke/module/model/HolE.html#HolE>`_ 得到。

SimplE
---------

`OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 实现的 `SimplE <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/SimplE.py>`__ 存在问题。
下面是 `SimplE <https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html>`__ 的作者给出的声明：

.. Important::

    Hi all, I'm the main author of the SimplE paper. I have received emails asking me if the OpenKE implementation of SimplE is correct or not so I thought I post a public response here. I can confirm that the OpenKE implementation is indeed incorrect and there are two issues (one major, one minor) in it:
    
    **Major issue**: As pointed out by @dschaehi there's a major issue in the model definition. SimplE requires two embedding vectors per entity, one to be used when the entity is the head and one to be used when the entity is the tail. In the OpenKE implementation, there is only one embedding vector per entity which hurts the model by making it almost identical to DistMult.
    
    **Minor issue**: This implementation corresponds to a variant of SimplE which we called SimplE-ignr in the paper. It takes the average of the two predictions during training but only uses one of the predictions during testing (see https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/SimplE.py#L54). The standard SimplE model takes the average of the two predictions for both training and testing.

    For a correct pytorch implementation of SimplE, I recommend this repo: https://github.com/baharefatemi/SimplE/blob/master/SimplE.py

关于这个问题的讨论在：https://github.com/thunlp/OpenKE/issues/151 。

因此，遵从 ``SimplE`` 作者的建议，依据 https://github.com/baharefatemi/SimplE/blob/master/SimplE.py 实现 ``SimplE`` 。

最终实现可以从 `这里 <_modules/pybind11_ke/module/model/SimplE.html#SimplE>`_ 得到。