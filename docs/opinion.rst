见解
=====

Analogy
---------

我去掉了原始 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 的 `Analogy` 的
`_calc <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
负号，原因如下：

在旧版的 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)>`__ 中，
`DistMult <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/DistMult.py#L23>`__、
`ComplEx <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/ComplEx.py#L36>`__、
`Analogy <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch(old)/models/Analogy.py#L30>`__ 3 者的
`_calc` 函数都带了负号，并且在
`Analogy 原论文的实现 <https://github.com/quark0/ANALOGY>`__ 中，
`DistMult <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L487>`__、
`ComplEx <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L527>`__、
`Analogy <https://github.com/quark0/ANALOGY/blob/master/main.cpp#L583>`__ 3 者的
`score` 函数都未带符号。从原论文中也能发现，三者的评分函数的符号应该是一致的。
但是在新版的 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 中，
三者 `DistMult <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/DistMult.py#L40>`__、
`ComplEx <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/ComplEx.py#L21>`__、
`Analogy <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
`_calc` 函数实现中，仅仅 `Analogy` 带了负号。

因此，我最终决定去掉 `OpenKE-PyTorch <https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch>`__ 的 `Analogy` 的
`_calc <https://github.com/LuYF-Lemon-love/pybind11-OpenKE/blob/thunlp-OpenKE-PyTorch/openke/module/model/Analogy.py#L27>`__ 的
负号。
