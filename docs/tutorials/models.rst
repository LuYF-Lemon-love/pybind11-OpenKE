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

:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 超参数，:math:`d` 是 :math:`L_{1}-norm` 或 :math:`L_{2}-norm`，

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

:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 超参数，:math:`C` 是一个加权软约束的超参数。

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
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 超参数。

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
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 超参数。

.. Important:: 为了加速收敛和避免过拟合，实体和关系的嵌入向量初始化为 TransE 的结果。

pybind11-OpenKE 的 TransD 实现传送门：:py:class:`pybind11_ke.module.model.TransD`

.. _rotate:

RotatE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RotatE`` :cite:`RotatE` 发表于 ``2019`` 年，将实体和关系映射到复数向量空间，并将每个关系定义为从头实体到尾实体的旋转。

欧拉恒等式 :math:`e^{i\theta}=\operatorname{cos}\theta + i\operatorname{sin}\theta` 表明酉复数（unitary complex number）可以看作是复平面中的旋转。

评分函数如下：

.. math::

    f_r(h,t)=\gamma - \Vert \mathbf{h} \circ \mathbf{r} - \mathbf{t} \Vert_{L_1}

:math:`\gamma` 是一个 **margin** 超参数，:math:`h, r, t \in \mathbb{C}^n` 是复数向量，:math:`|r_i|=1`，:math:`\circ` 表示哈达玛积。对于复数向量空间中的每一维度，``RotatE`` :cite:`RotatE` 假设：

.. math::

    t_i = h_i r_i, \text{ where } h_i, r_i, t_i \in \mathbb{C} \text{ and } |r_i|=1. 

事实证明，这种简单的操作可以有效地模拟所有三种关系模式：对称/非对称（symmetry/antisymmetry）、反转（ inversion）和组合（composition）。

损失函数如下：

.. math::

    \mathcal{L} = -\log\sigma(f_r(h,t))-\sum_{i=1}^{n}\frac{1}{n}\log\sigma(-f_r(h_i^{'},t_i^{'}))
    
:math:`\sigma` 表示 sigmoid 函数。

由于均匀的负采样存在效率低下的问题，因为随着训练的进行，许多样本显然是假的，这不能提供任何有意义的信息。因此，``RotatE`` :cite:`RotatE` 提出了一种称为自对抗负采样（self-adversarial negative sampling）的方法，该方法根据当前的嵌入模型对负三元组进行采样。具体来说，从以下分布中采样负三元组：

.. math::

    p_r(h_j^{'},t_j^{'}|\{(h,r,t)\})=\frac{\operatorname{exp}af_r(h_j^{'},t_j^{'})}{\sum_i\operatorname{exp}af_r(h_i^{'},t_i^{'})}

其中 :math:`a` 是采样的温度。将上述概率视为负样本的权重，损失函数变为：

.. math::

    \mathcal{L} = -\log\sigma(f_r(h,t))-\sum_{i=1}^{n}p_r(h_i^{'},t_i^{'})\log\sigma(-f_r(h_i^{'},t_i^{'}))

pybind11-OpenKE 的 RotatE 实现传送门：:py:class:`pybind11_ke.module.model.RotatE`

语义匹配模型
----------------------------------

.. _rescal:

RESCAL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RESCAL`` :cite:`RESCAL` 发表于 ``2011`` 年，是 ``DistMult`` :cite:`DistMult` 的基石，即没有限制关系矩阵 :math:`M_r` 为对角矩阵。

评分函数如下：

.. math::

    f_r(h,t)=\mathbf{h}^T \mathbf{M}_r \mathbf{t}

:math:`\mathbf{M}_r` 是关系 :math:`r` 对应的关系矩阵。

pybind11-OpenKE 的 RESCAL 实现传送门：:py:class:`pybind11_ke.module.model.RESCAL`

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

其中循环相关算子 :math:`\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d` 定义为：

.. math::
    
    [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} \textbf{b}_{(i+k)\ mod \ d}

通过使用循环相关运算符，:math:`[\textbf{h} \star \textbf{t}]_i` 每个分量在成对交互中表示固定分区的总和。这使模型能够将语义相似的交互放入同一分区中，并通过 :math:`\textbf{r}` 共享权重。同样，不相关的特征交互也可以放在同一个分区中，该分区可以在 :math:`\textbf{r}` 中分配较小的权重。

可以通过快速傅里叶变换（fast Fourier transform，FFT）实现循环相关算子，进而评分函数可以表示为如下形式：

.. math::
    
    f_r(h,t)=\mathbf{r}^T (\mathcal{F}^{-1}(\overline{\mathcal{F}(\mathbf{h})} \odot \mathcal{F}(\mathbf{t})))

其中 :math:`\mathcal{F}(\cdot)` 和 :math:`\mathcal{F}^{-1}(\cdot)` 表示快速傅里叶变换，:math:`\overline{\mathbf{x}}` 表示复数共轭，:math:`\odot` 表示哈达玛积。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    [\gamma + f_r(h^{'},t^{'}) - f_r(h,t)]_{+}
    
:math:`[x]_{+}` 表示 :math:`x` 的正数部分，:math:`\gamma > 0` 是一个 **margin** 超参数。

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

pybind11-OpenKE 的 ComplEx 实现传送门：:py:class:`pybind11_ke.module.model.ComplEx`

.. _analogy:

ANALOGY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ANALOGY`` :cite:`ANALOGY` 发表于 ``2017`` 年，是一个显式地建模类比结构的模型；但实际上是 ``DistMult`` :cite:`DistMult`、 ``HolE`` :cite:`HolE` 和 ``ComplEx`` :cite:`ComplEx` 的集大成者，效果与 ``HolE`` :cite:`HolE` 和 ``ComplEx`` :cite:`ComplEx` 差不多。

当且仅当 :math:`A^TA = AA^T`，实矩阵 :math:`A` 是正规的（normal）。

评分函数如下：

.. math::

    f_r(h,t)=\mathbf{h}^T \mathbf{M}_r \mathbf{t}

:math:`\mathbf{M}_r` 是关系 :math:`r` 对应的关系矩阵。

为了显式地建模类比结构，:math:`\mathbf{M}_r` 还需要满足下面的约束条件（分别为正规性和交换性）：
	
.. math::
    
    \mathbf{M}_r\mathbf{M}_r^T = \mathbf{M}_r^T\mathbf{M}_r\\
    \mathbf{M}_r\mathbf{M}_{r^{'}} = \mathbf{M}_{r^{'}}\mathbf{M}_r

直接优化上面的模型需要大量的计算，经过作者的推理发现，:math:`\mathbf{M}_r` 是块对角矩阵（a block-diagonal matrix），:math:`\mathbf{M}_r` 的每个对角块是下面两种情况之一：

- 一个实数标量（real scalar）；
- :math:`\begin{bmatrix} x & -y \\y & x \end{bmatrix}` 形式的二维实数矩阵，:math:`x` 和 :math:`y` 都是实数标量。

通过将 :math:`\mathbf{M}_r` 的实数标量和二维实数矩阵的系数各自绑定到一起：

- 实数标量绑定到一起会形成一个对角矩阵，如同 ``DistMult`` :cite:`DistMult` 的关系矩阵一样。
- 二维实数矩阵绑定到一起会形成一个类似 ``ComplEx`` :cite:`ComplEx` 的关系矩阵，原因如下：第 :math:`i` 块可以表示为 :math:`\begin{bmatrix} \operatorname{Re}(r) & -\operatorname{Im}(r) \\ \operatorname{Im}(r) & \operatorname{Re}(r) \end{bmatrix}`，如果实体也是复数向量，这一部分的得分函数 :math:`f_r(h,t)=\mathbf{h}^T \mathbf{M}_r \mathbf{t}` 的计算结果会和 ``ComplEx`` :cite:`ComplEx` 的得分函数一样。

在原论文中，实数标量和二维实数矩阵的维度相同，即各占关系矩阵一半的维度。因此，最终的评分函数实际上是 ``DistMult`` :cite:`DistMult` 评分函数和 ``ComplEx`` :cite:`ComplEx` 评分函数的和：

.. math::

    f_r(h,t)=<\operatorname{Re}(\mathbf{h_c}),\operatorname{Re}(\mathbf{r_c}),\operatorname{Re}(\mathbf{t_c})>
             +<\operatorname{Re}(\mathbf{h_c}),\operatorname{Im}(\mathbf{r_c}),\operatorname{Im}(\mathbf{t_c})>
             +<\operatorname{Im}(\mathbf{h_c}),\operatorname{Re}(\mathbf{r_c}),\operatorname{Im}(\mathbf{t_c})>
             -<\operatorname{Im}(\mathbf{h_c}),\operatorname{Im}(\mathbf{r_c}),\operatorname{Re}(\mathbf{t_c})>
             +<\mathbf{h_d}, \mathbf{r_d}, \mathbf{t_d}>

:math:`h_c, r_c, t_c` 是 ``ComplEx`` :cite:`ComplEx` 部分对应的头实体、关系和尾实体的嵌入向量，:math:`h_d, r_d, t_d` 是 ``DistMult`` :cite:`DistMult` 部分对应的头实体、关系和尾实体的嵌入向量。

损失函数如下：

.. math::

    \mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h^{'},r,t^{'}) \in S^{'}_{(h,r,t)}}
    \log(1+exp(-yf_r(h,t)))+\lambda\Vert \theta \Vert^2_2

:math:`\theta` 是模型的参数。

pybind11-OpenKE 的 ANALOGY 实现传送门：:py:class:`pybind11_ke.module.model.Analogy`

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

图神经网络模型
----------------------------------

.. _rgcn:

R-GCN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``R-GCN`` :cite:`R-GCN` 发表于 ``2017`` 年，本质是一个编码器。在链接预测时，``R-GCN`` 将会生成实体的潜在特征表示；然后利用 ``DistMult`` :cite:`DistMult` 生成三元组的得分。

回顾一下 GCN 模型，第 :math:`(l+1)` 层的节点 :math:`i` 的隐藏表示计算如下（消息传递范式）：

.. math::

    h_i^{(l+1)} = \sigma\left(\sum_{m \in M_i}g_m ( h_i^{(l)}, h_j^{(l)}) \right)~~~~~~~~~~(1)\\

:math:`h_i^{(l)} \in \mathbb{R}^{d^{(l)}}` 是图神经网络第 :math:`l` 层节点 :math:`v_i` 的隐藏状态，其中维度为 :math:`d^{(l)}`。:math:`g_m(.,.)` 是定义在每条边上的消息函数，上面的公式使用 ``sum`` 作为聚合函数，:math:`M_i` 表示节点 :math:`v_i` 的传入消息集合（the set of incoming messages），并且通常被选择为与传入边集合（the set of incoming edges）相同。消息函数 :math:`g_m(.,.)` 可以是简单的一元函数或者二元函数，如 ``copy``, ``add``, ``sub``, ``mul``, ``div``, ``dot``；也可以是权重为 :math:`W` 的线性变换 :math:`g_m(h_i, h_j) = Wh_j`。:math:`\sigma` 是一个激活函数。

R-GCN 模型中第 :math:`(l+1)` 层的节点 :math:`i` 的隐藏表示计算如下：

.. math::

    h_i^{(l+1)} = \sigma\left(W_0^{(l)}h_i^{(l)}+\sum_{r\in R}\sum_{j\in N_i^r}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}\right)~~~~~~~~~~(2)\\

其中 :math:`N_i^r` 表示关系 :math:`r \in R` 下节点 :math:`i` 的邻居索引集合。:math:`c_{i,r}` 是归一化常数，R-GCN 论文使用 :math:`c_{i,r}=|N_i^r|`。为了确保第 :math:`l + 1`` 层节点的表示能够获悉第 :math:`l` 层的相应表示，作者为数据中的每个节点添加一个特殊关系类型（self-connection），:math:`W_0` 是自循环权重。

.. figure:: /_static/images/tutorials/RGCN01.png
    :align: center

为了防止过拟合，作者提出了两种方法正则化 R-GCN 层的权重：

1. 基础正则化（The basis regularization）分解 :math:`W_r` 为：

.. math::

    W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}~~~~~~~~~~(3)\\

其中 :math:`B` 是基矩阵的个数，:math:`a_{rb}^{(l)}` 是取决于关系 :math:`r` 的系数，基矩阵为 :math:`V_b^{(l)} \in \mathbb{R}^{d^{(l+1)} \times d^{(l)}}`。

2. 块对角线分解正则化（The block-diagonal-decomposition regularization）将 :math:`W_r` 分解为 :math:`B` 个块对角矩阵：

.. math::

    W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}~~~~~~~~~~(4)\\

:math:`Q_{rb}^{(l)} \in \mathbb{R}^{(d^{(l+1)}/B) \times (d^{(l)}/B)}`，:math:`W_r^{(l)}` 是块对角矩阵：:math:`\operatorname{diag}(Q_{r1}^{(l)},...,Q_{rB}^{(l)})`。

基础正则化（3）可以看作是不同关系类型之间有效权重共享的一种形式，而块对角线分解正则化（4）可以看作对每个关系类型的权重矩阵的稀疏性约束。

.. figure:: /_static/images/tutorials/RGCN02.png
    :align: center

链接预测时，R-GCN 作为编码器输出实体的表示，关系的表示来自于 ``DistMult`` 模型。损失函数为 :py:class:`torch.nn.BCEWithLogitsLoss`。

pybind11-OpenKE 的 RGCN 实现传送门：:py:class:`pybind11_ke.module.model.RGCN`

.. _compgcn:

CompGCN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CompGCN`` :cite:`CompGCN` 发表于 ``2020`` 年，这是一种在图卷积网络中整合多关系信息的新框架，它利用知识图谱嵌入技术中的各种组合操作，将实体和关系共同嵌入到图中。

通过增加反向边（逆关系）对知识图谱的有向边（关系）进行扩展，使得有向边的信息可以双向流动：

.. math::

    S^{'}_{(h,r,t)}=S_{(h,r,t)} \cup \{ (t,r^{-1},h) | (h,r,t) \in S_{(h,r,t)} \} \cup \{ (h, T, h) | h \in E \}~~~~~~~~~~(1)\\

其中 :math:`S_{(h,r,t)}` 表示知识图谱的所有三元组，:math:`R^{'} = R \cup R_{inv} \cup T`，:math:`R_{inv} = \{r^{-1} | r \in R \}` 表示逆关系，:math:`T` 表示自循环关系。

使用了减法（来自于 ``TransE`` :cite:`TransE` ）、乘法（来自于 ``DistMult`` :cite:`DistMult` ）、循环相关（来自于 ``HolE`` :cite:`HolE` ）三种知识图谱嵌入组合操作将关系融合到尾实体的信息中，进而使用图神经网络进行编码：

.. math::

    t = \phi\left( h, r \right)~~~~~~~~~~(2)\\

:math:`h, r, t` 分表示头实体，关系，尾实体的嵌入向量，:math:`\phi : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d` 表示组合操作计算方式如下：

.. math::

    \phi\left( h, r \right) = \begin{cases}
                            h-r, & \text{Subtraction (Sub)} \\
                            \sum_{i=1}^{n}h_ir_i, & \text{Multiplication (Mult)} \\
                            h \star r, & \text{Circular-correlation (Corr)}
                            \end{cases}~~~~~~~~~~(3)\\

图神经网络尾实体隐藏表示计算如下：

.. math::

    t = \sigma\left( \sum_{(h,r) \in N_t} W_{\lambda}\phi\left( h, r \right)\right)~~~~~~~~~~(4)\\

其中 :math:`N_t` 表示尾实体 :math:`t` 的头实体关系集合。:math:`W_{\lambda}` 是一个特定于关系类型的参数（原始的关系，逆关系，自循环关系）。

.. math::

    W_{\lambda} = \begin{cases}
                W_O, & r \in R \\
                W_I, & r \in R_{inv} \\
                W_S, & r = T
                \end{cases}~~~~~~~~~~(5)\\

在实体的嵌入向量更新完后，需要对关系嵌入向量进行更新：

.. math::

    r^{'} = W_{r}r~~~~~~~~~~(6)\\

:math:`W_r` 是线性变换矩阵，:math:`r^{'}` 表示更新后的关系嵌入向量。

受 ``R-GCN`` :cite:`R-GCN` 启发，作者对关系嵌入向量进行了正则化：

.. math::

    r = \sum_{b=1}^B a_{br}v_b~~~~~~~~~~(7)\\

其中，基向量为 :math:`v_b`，:math:`B` 是基向量的个数，:math:`a_{br}` 是特定于关系和基向量的可学习的标量权重。仅仅第一层使用上述的正则化，其他层的关系向量来源于公式 6 更新后的关系向量。

因此，图神经网络的每一层的更新公式如下：

.. math::

    t^{l+1} = \sigma\left( \sum_{(h,r) \in N_t} W^{l}_{\lambda}\phi\left( h^{l}, r^{l} \right)\right)~~~~~~~~~~(8)\\

设 :math:`t^{l+1}` 表示在 :math:`l` 层之后获得的尾实体 :math:`t` 的表示。相似的，:math:`r^{l+1}` 表示 :math:`l` 层之后关系 :math:`r` 的表示：

.. math::

    r^{l+1} = W^{l}_{r}r^{l}~~~~~~~~~~(9)\\

链接预测时，``CompGCN`` :cite:`CompGCN` 也是作为编码器输出实体和关系的表示，然后用传统的知识图谱嵌入模型进行解码，原论文使用如下 3 种知识图谱嵌入模型作为解码器：``TransE`` :cite:`TransE`，``DistMult`` :cite:`DistMult` 和 ``ConvE``。其中使用循环相关操作符（Circular-correlation (Corr)） 和 ``ConvE`` 作为解码器的组合在论文中取得了最好的效果。

对于训练链接预测模型，使用带有标签平滑的标准二元交叉熵损失，损失函数为 :py:class:`torch.nn.BCELoss`。

pybind11-OpenKE 的 CompGCN 实现传送门：:py:class:`pybind11_ke.module.model.CompGCN`