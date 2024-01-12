实现细节
==================================

SimplE
---------

`OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 实现的 `SimplE <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/SimplE.py>`__ 存在问题。
下面是 ``SimplE`` :cite:`SimplE` 的作者给出的声明：

.. Important::

    Hi all, I'm the main author of the SimplE paper. I have received emails asking me if the OpenKE implementation of SimplE is correct or not so I thought I post a public response here. I can confirm that the OpenKE implementation is indeed incorrect and there are two issues (one major, one minor) in it:
    
    **Major issue**: As pointed out by @dschaehi there's a major issue in the model definition. SimplE requires two embedding vectors per entity, one to be used when the entity is the head and one to be used when the entity is the tail. In the OpenKE implementation, there is only one embedding vector per entity which hurts the model by making it almost identical to DistMult.
    
    **Minor issue**: This implementation corresponds to a variant of SimplE which we called SimplE-ignr in the paper. It takes the average of the two predictions during training but only uses one of the predictions during testing (see https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/SimplE.py#L54). The standard SimplE model takes the average of the two predictions for both training and testing.

    For a correct pytorch implementation of SimplE, I recommend this repo: https://github.com/baharefatemi/SimplE/blob/master/SimplE.py

关于这个问题的讨论在：https://github.com/thunlp/OpenKE/issues/151 。

因此，遵从 ``SimplE`` 作者的建议，依据 https://github.com/baharefatemi/SimplE/blob/master/SimplE.py 实现 ``SimplE`` 。

最终实现可以从 `这里 <_modules/pybind11_ke/module/model/SimplE.html#SimplE>`_ 得到。

.. _details_hole:

HolE
---------

.. WARNING:: 由于 :py:class:`pybind11_ke.module.model.HolE` 的
    :py:meth:`pybind11_ke.module.model.HolE._ccorr` （`OpenKE-PyTorch 的原始实现 <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/HolE.py#L60>`__）需要
    `torch.rfft <https://pytorch.org/docs/1.7.0/generated/torch.rfft.html#torch.rfft>`_ 和 `torch.ifft <https://pytorch.org/docs/1.7.0/generated/torch.ifft.html#torch.ifft>`_ 分别计算实数到复数离散傅里叶变换和复数到复数离散傅立叶逆变换。
    ``pytorch`` 在版本 ``1.8.0`` 移除了上述两个函数，并且在版本 ``1.7.0`` 给出了警告。
    因此，需要适配到更高版本的 ``pytorch``。

.. Important::
    我参考了 `PyKEEN 的 hole_interaction 实现 <https://pykeen.readthedocs.io/en/stable/api/pykeen.nn.functional.hole_interaction.html#pykeen.nn.functional.hole_interaction>`_ ，重新实现了 :py:class:`pybind11_ke.module.model.HolE`，
    使其能够适配到更高版本的 ``pytorch``。

RESCAL
---------

我去掉了原始 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 的 ``RESCAL`` 的
`predict <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/RESCAL.py#L45>`__ 的
负号，原因如下：

由于 :py:class:`pybind11_ke.module.model.RESCAL` 采用 :py:class:`pybind11_ke.module.loss.MarginLoss` 进行训练，因此需要正样本评分函数的得分应小于负样本评分函数的得分，
因此，:py:class:`pybind11_ke.module.model.RESCAL` 的评分函数需要添加负号即 :py:class:`pybind11_ke.module.model.RESCAL.forward` 需要添加符号；
由于 pybind11-OpenKE 使用的底层 C++ 模块进行评估模型性能，该模块需要正样本的得分小于负样本的得分，
因此 :py:class:`pybind11_ke.module.model.RESCAL.predict` 不需要在 :py:class:`pybind11_ke.module.model.RESCAL.forward` 返回的结果上添加负号。

.. Important::
    实验表明，去掉负号能够大幅度改善模型的评估结果。

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