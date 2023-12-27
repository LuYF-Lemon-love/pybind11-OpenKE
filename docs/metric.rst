评估指标
==================================

对于每一个测试三元组，头实体依次被实体集中的每一个实体替换。
这些损坏的三元组首先被 KGE 模型计算得分，然后按照得分排序。
我们因此得到了正确实体的排名。上述过程通过替换尾实体被重复。
最后，报道了在上述设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10, Hits@3, Hits@1），
平均排名（mean rank，MR），平均倒数排名（mean reciprocal rank，MRR）。

因为一些损坏的三元组可能已经存在训练集和验证集中，在这种情形下，
这些损坏三元组可能比测试集中的三元组排名更靠前，但不应该被认为是
错误的，因为两个三元组都是正确的。我们删除了那些出现在训练集、验证集或测试集中的损坏的三元组，
这确保了这样的损坏三元组不会参与排名。最后，报告了在这种设置上测试集三元组的正确实体排名在前 10/3/1 的比例（Hits@10 (filter), Hits@3(filter), Hits@1(filter)），
平均排名（mean rank，MR(filter)），平均倒数排名（mean reciprocal rank，MRR(filter)）。

上述设置更多的细节可以从 ``TransE`` :cite:`TransE` 获得。

对于大型知识图谱，用整个实体集合损坏三元组是极其耗时的。
因此提供了名为
"`type constraint <https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf>`__"
的实验性的设置用有限的实体集合（取决于相应的关系）损坏三元组。