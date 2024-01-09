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

.. _transd:

TransD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransD`` :cite:`TransD` 发表于 ``2015`` 年，是 ``TransR`` :cite:`TransR` 的改进版，为实体和关系分别定义了两个向量。第一个向量表示实体或关系的意义；另一个向量（投影向量）表示如何将实体嵌入向量投影到关系向量空间，投影向量被用来构建映射矩阵。因此，每个实体-关系对有独一无二的映射矩阵。

评分函数如下：

.. math::

    f_r(h,t)=\Vert (\mathbf{r}_p \mathbf{h}_p^T + \mathbf{I})\mathbf{h} + \mathbf{r} - (\mathbf{r}_p \mathbf{t}_p^T + \mathbf{I})\mathbf{t} \Vert^2_2

对于三元组 :math:`(h, r, t)`，:math:`h,r,t` 分别表示头实体、关系和尾实体的嵌入向量，:math:`h_p,r_p,t_p` 分别表示头实体、关系和尾实体的投影向量，:math:`I` 表示单位矩阵。

除此之外还有下面的约束条件：
	
.. math::
    
    \Vert \mathbf{h} \Vert_2 \leq 1,\\
    \Vert \mathbf{r} \Vert_2 \leq 1,\\
    \Vert \mathbf{t} \Vert_2 \leq 1,\\
    \Vert (\mathbf{r}_p \mathbf{h}_p^T + \mathbf{I})\mathbf{h} \Vert_2 \leq 1,\\
    \Vert (\mathbf{r}_p \mathbf{t}_p^T + \mathbf{I})\mathbf{t} \Vert_2 \leq 1.

.. Important:: 实体和关系嵌入向量的维度不需要相同。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + f_r(h,t) - f_r(h^{'},t^{'})]_{+}
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 函数。

.. Important:: 为了加速收敛和避免过拟合，实体和关系的嵌入向量初始化为 TransE 的结果。

pybind11-OpenKE 的 TransD 实现传送门：:py:class:`pybind11_ke.module.model.TransD`

语义匹配模型
----------------------------------

.. _distMult:

DistMult
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DistMult`` :cite:`DistMult` 发表于 ``2015`` 年，是一个简单的双线性模型，限制关系矩阵 :math:`M_r` 为对角矩阵。

评分函数如下：

.. math::

    f_r(h,t)=\sum_{i=1}^{n}h_ir_it_i

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [1 + f_r(h^{'},t^{'}) - f_r(h,t)]_{+}

.. Important:: 原论文为训练集中的每一个三元组构建了 2 个负三元组：一个通过替换头实体得到的和一个通过替换尾实体得到的。每次更新参数后，实体向量被重新规范为单位向量。对关系向量施加了 :math:`L_2` 正则化。

.. Important:: DistMult 不能够区分关系 :math:`r` 和与关系 :math:`r` 相反的关系。

pybind11-OpenKE 的 DistMult 实现传送门：:py:class:`pybind11_ke.module.model.DistMult`

.. _hole:

HolE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``HolE`` :cite:`HolE` 发表于 ``2016`` 年，全息嵌入（HolE）利用循环相关算子来计算实体和关系之间的交互。

评分函数如下：

.. math::

    f_r(h,t)= \sigma(\textbf{r}^{T}(\textbf{h} \star \textbf{t}))

其中循环相关算子 $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ 定义为：

.. math::
    
    [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} \textbf{b}_{(i+k)\ mod \ d}

通过使用循环相关运算符，$[\textbf{h} \star \textbf{t}]_i$ 每个分量在成对交互中表示固定分区的总和。这使模型能够将语义相似的交互放入同一分区中，并通过 $\textbf{r}$ 共享权重。同样，不相关的特征交互也可以放在同一个分区中，该分区可以在 $\textbf{r}$ 中分配较小的权重。

可以通过快速傅里叶变换（fast Fourier transform，FFT）实现循环相关算子，进而评分函数可以表示为如下形式：

.. math::
    
    f_r(h,t)=\mathbf{r}^T (\mathcal{F}^{-1}(\overline{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{t})))

其中 :math:`\mathcal{F}(\cdot)` 和 :math:`\mathcal{F}^{-1}(\cdot)` 表示快速傅里叶变换，:math:`\overline{\mathbf{x}}` 表示复数共轭，:math:`\odot` 表示哈达玛积。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + f_r(h^{'},t^{'}) - f_r(h,t)]_{+}
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 函数。

pybind11-OpenKE 的 HolE 实现传送门：:py:class:`pybind11_ke.module.model.HolE`

.. _complex:

ComplEx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ComplEx`` :cite:`ComplEx` 发表于 ``2016`` 年，是一个复数版本的 DistMult，利用复数共轭建模非对称关系。

评分函数如下：

.. math::

    f_r(h,t)=<\operatorname{Re}(h),\operatorname{Re}(r),\operatorname{Re}(t)>
             +<\operatorname{Re}(h),\operatorname{Im}(r),\operatorname{Im}(t)>
             +<\operatorname{Im}(h),\operatorname{Re}(r),\operatorname{Im}(t)>
             -<\operatorname{Im}(h),\operatorname{Im}(r),\operatorname{Re}(t)>

:math:`h, r, t \in \mathbb{C}^n` 是复数向量，:math:`< \mathbf{a}, \mathbf{b}, \mathbf{c} >=\sum_{i=1}^{n}a_ib_ic_i` 为逐元素多线性点积（element-wise multi-linear dot product）。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    \log(1+exp(-yf_r(h,t)))+\lambda\Vert \theta \Vert^2_2

:math:`\theta` 是模型的参数。

.. Important:: 对数似然损失（log-likelihood loss）比成对排名损失（pairwise ranking loss）效果更好；每一个训练三元组生成更多的负三元组会产生更好的效果。

pybind11-OpenKE 的 DistMult 实现传送门：:py:class:`pybind11_ke.module.model.ComplEx`

.. _simple:

SimplE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SimplE`` :cite:`SimplE` 发表于 ``2018`` 年，是一个双线性模型，为每一个实体构建了 2 个向量：:math:`h_e` 和 :math:`t_e`，为每一个关系构建了 2 个向量：:math:`r` 和 :math:`r^{-1}`。

评分函数如下：

.. math::

    f_r(h,t)=\frac{1}{2}(\sum_{i=1}^{n}h_{hi}r_it_{ti}+\sum_{i=1}^{n}h_{ti}r^{-1}_it_{hi})

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    \log(1+exp(-yf_r(h,t)))+\lambda\Vert \theta \Vert^2_2

:math:`\theta` 是模型的参数。

.. Important:: 平均倒数排名（mean reciprocal rank，MRR(filter)）比平均排名（mean rank，MR(filter)）更具有鲁棒性，由于仅仅 1 个坏的 rank 能够很大的影响 MR。

pybind11-OpenKE 的 DistMult 实现传送门：:py:class:`pybind11_ke.module.model.SimplE`
