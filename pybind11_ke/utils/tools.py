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

    """从模块中导入类。"""

    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_