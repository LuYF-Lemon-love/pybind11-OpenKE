知识图谱嵌入模型
==================================

平移模型
----------------------------------

.. _transe:

TransE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransE`` :cite:`TransE` 发表于 ``2013`` 年，是一个将关系建模为实体低维向量的平移操作的模型。如果 :math:`(h,r,t)` 成立，尾实体 :math:`t` 的向量应该接近头实体 :math:`h` 的向量加上关系 :math:`r` 的向量。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + d(h+r,t) - d(h^{'}+r,t^{'})]_{+}

:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 函数，:math:`d` 是 :math:`L_{1}-norm` 或 :math:`L_{2}-norm`，

.. math::

    S^{'}_{(h,r,t)}=\{(h^{'},r,t)|h^{'} \in E\} \cup \{(h,r,t^{'})|t^{'} \in E\}

负三元组是由训练集三元组根据上面的公式随机替换头实体或尾实体得到的（不同时）。

.. Important:: 对于给定的实体，作为头实体和尾实体时，它的嵌入向量是相同的。原论文中对实体向量施加了 :math:`L_{2}-norm` 限制，将实体限制为 1。

pybind11-OpenKE 的 TransE 实现传送门：:py:class:`pybind11_ke.module.model.TransE`

.. _transh:

TransH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransH`` :cite:`TransH` 发表于 ``2014`` 年，是一个将关系建模为实体低维向量在超平面上的平移操作的模型。每一个关系 :math:`r` 被 2 个向量表示：超平面的的法向量 :math:`r_w` 和超平面上的平移向量 :math:`r_d`。如果 :math:`(h,r,t)` 成立，尾实体 :math:`t` 在超平面上的投影向量应该接近头实体 :math:`h` 在超平面上的投影向量加上关系超平面上的平移向量 :math:`r_d`。

评分函数如下：

.. math::

    f_r(h,t)=\Vert (h-r_w^\top hr_w)+r_d-(t-r_w^\top tr_w)\Vert^2_2

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + f_r(h,t) - f_r(h^{'},t^{'})]_{+}+
    C\Bigg\{ \sum_{e \in E}[\Vert e \Vert^2_2 - 1]_{+} + \sum_{r \in R}\bigg[ \frac{(r_w^\top r_d)^2}{\Vert r_d \Vert^2_2} - \epsilon^2\bigg]_{+} \Bigg\}

:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 函数，:math:`C` 是一个加权软约束的超参数。

.. Important:: 在每次 batch 迭代前，需要将 :math:`r_w` 投影到 :math:`l_2` 单位球上保证 :math:`r_w` 是单位法向量。

除此之外，该论文还提出一个负采样 **trick** 来降低生成假负（false negative）样本的机会：如果关系是 one-to-many 的话，更多地通过替换头实体来生成负样本；如果关系是 many-to-one 的话，更多地通过替换尾实体来生成负样本。关系 one-to-many 和关系 many-to-one 更多的细节可以从 ``TransE`` :cite:`TransE` 的原论文获得。下面是该 trick 的工作流程：

1. 对于给定关系 :math:`r` 的所有三元组，首先从训练集中得到两种统计信息：计算每个头实体尾实体的平均数量，记为 :math:`tph`；计算每个尾实体头实体的平均数量，记为 :math:`hpt`。

2. 对于训练集中的 :math:`(h,r,t)` 三元组，我们以 :math:`\frac{tph}{tph+hpt}` 概率替换头实体构造负三元组；以 :math:`\frac{hpt}{tph+hpt}` 概率替换尾实体构造负三元组。

由于上面负采样过程中定义了一个伯努利分布（Bernoulli distribution），所以该采样方法被记为 ``bern.``，TransE 中以均等概率替换头尾实体的采样方法被记为 ``unif``。

pybind11-OpenKE 的 TransH 实现传送门：:py:class:`pybind11_ke.module.model.TransH`

.. _transr:

TransR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransR`` :cite:`TransR` 发表于 ``2015`` 年，是一个为实体和关系嵌入向量分别构建了独立的向量空间，将实体向量投影到特定的关系向量空间进行平移操作的模型。如果 :math:`(h,r,t)` 成立，首先使用特定关系矩阵 :math:`M_r` 将头尾实体向量投影到特定关系 :math:`r` 的向量空间：:math:`h_r` 和 :math:`t_r`，然后尾实体投影后的向量 :math:`t_r` 应该接近头实体投影后的向量 :math:`h_r` 加上关系的向量 :math:`r`。

评分函数如下：

.. math::

    f_r(h,t)=\Vert hM_r+r-tM_r \Vert^2_2

除此之外还有下面的约束条件：

.. math::
    
    \Vert h \Vert_2 \leq 1,\\
    \Vert r \Vert_2 \leq 1,\\
    \Vert t \Vert_2 \leq 1,\\
    \Vert hM_r \Vert_2 \leq 1,\\
    \Vert tM_r \Vert_2 \leq 1.

.. Important:: 实体和关系嵌入向量的维度不需要相同。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + f_r(h,t) - f_r(h^{'},t^{'})]_{+}
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 函数。

.. Important:: 为了避免过拟合，实体和关系的嵌入向量初始化为 TransE 的结果，关系矩阵 :math:`M_r` 初始为单位矩阵。

pybind11-OpenKE 的 TransR 实现传送门：:py:class:`pybind11_ke.module.model.TransR`

语义匹配模型
----------------------------------

.. _distMult:

DistMult
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DistMult`` :cite:`DistMult` 发表于 ``2015`` 年，是一个简单的双线性模型，限制关系矩阵 :math:`M_r` 为对角矩阵。

评分函数如下：

.. math::

    f_r(h,t)=<h,r,t>=\sum_{i=1}^{n}h_ir_it_i

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [1 + f_r(h^{'},t^{'}) - f_r(h,t)]_{+}

.. Important:: 原论文为训练集中的每一个三元组构建了 2 个负三元组：一个通过替换头实体得到的和一个通过替换尾实体得到的。每次更新参数后，实体向量被重新规范为单位向量。对关系向量施加了 :math:`L_2` 正则化。

.. Important:: DistMult 不能够区分关系 :math:`r` 和与关系 :math:`r` 相反的关系。

pybind11-OpenKE 的 DistMult 实现传送门：:py:class:`pybind11_ke.module.model.DistMult`
