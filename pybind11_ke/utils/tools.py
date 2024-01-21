# coding:utf-8
#
# pybind11_ke/utils/tools.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 3, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 3, 2024
#
# 该脚本定义了 WandbLogger 类.

import importlib

def import_class(module_and_class_name: str) -> type:

    """从模块中导入类。
    
    :param module_and_class_name: 模块和类名，如 ``pybind11_ke.module.model.TransE`` 。
    :type module_and_class_name: str
    :returns: 类名
    :rtype: type
    """

    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def construct_type_constrain(
    in_path: str = "./",
    train_file: str = "train2id.txt",
    valid_file: str = "valid2id.txt",
    test_file: str = "test2id.txt"
    ):

    """构建 type_constrain.txt 文件
    
    type_constrain.txt: 类型约束文件, 第一行是关系的个数
    
    下面的行是每个关系的类型限制 (训练集、验证集、测试集中每个关系存在的 head 和 tail 的类型)
    
    每个关系有两行：
    
    第一行：``id of relation`` ``Number of head types`` ``head1`` ``head2`` ...
    
    第二行: ``id of relation`` ``number of tail types`` ``tail1`` ``tail2`` ...

    如 benchmarks/FB15K 的 id 为 1200 的关系，它有 4 种类型头实体（3123，1034，58 和 5733）和 4 种类型的尾实体（12123，4388，11087 和 11088）。
    
    1200	4	3123	1034	58	5733

    1200	4	12123	4388	11087	11088
    
    :param in_path: 数据集目录
    :type in_path: str
    :param train_file: train2id.txt
    :type train_file: str
    :param valid_file: valid2id.txt
    :type valid_file: str
    :param test_file: test2id.txt
    :type test_file: str
    """


    rel_head: dict = {}
    rel_tail: dict = {}
    
    train = open(in_path + train_file, "r")
    valid = open(in_path + valid_file, "r")
    test = open(in_path + test_file, "r")
    
    tot = (int)(train.readline())
    for i in range(tot):
        content = train.readline()
        h,t,r = content.strip().split()
        if not r in rel_head:
            rel_head[r] = {}
        if not r in rel_tail:
            rel_tail[r] = {}
        rel_head[r][h] = 1
        rel_tail[r][t] = 1
    
    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h,t,r = content.strip().split()
        if not r in rel_head:
            rel_head[r] = {}
        if not r in rel_tail:
            rel_tail[r] = {}
        rel_head[r][h] = 1
        rel_tail[r][t] = 1
        
    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h,t,r = content.strip().split()
        if not r in rel_head:
            rel_head[r] = {}
        if not r in rel_tail:
            rel_tail[r] = {}
        rel_head[r][h] = 1
        rel_tail[r][t] = 1
    
    train.close()
    valid.close()
    test.close()
    
    f = open(in_path + "type_constrain.txt", "w")
    f.write("%d\n" % (len(rel_head)))
    for i in rel_head:
        f.write("%s\t%d" % (i, len(rel_head[i])))
        for j in rel_head[i]:
            f.write("\t%s" % (j))
        f.write("\n")
        f.write("%s\t%d" % (i, len(rel_tail[i])))
        for j in rel_tail[i]:
            f.write("\t%s" % (j))
        f.write("\n")
    f.close()