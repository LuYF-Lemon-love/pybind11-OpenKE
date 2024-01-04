# coding:utf-8
#
# pybind11_ke/utils/Timer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 6, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 4, 2023
#
# 该脚本定义了计时器类.

"""
:py:class:`Timer` - 计时器类。
"""

import time

class Timer:

    """记录多次运行时间"""

    def __init__(self):

        """创建 Timer 对象。"""

        #: 存放时间间隔的列表
        self.times: list[float] = []
        #: 记录当前时间
        self.current: float = None
        #: 记录上一次的时间
        self.last: float = None

        self.restart()

    def restart(self):

        """重启计时器。"""

        self.last = self.current = time.time()

    def stop(self) -> float:

        """停止计时器并将时间记录在列表中。
        
        :returns: 返回最后一次的间隔时间。
        :rtype: float
        """
        
        self.current = time.time()
        self.times.append(self.current - self.last)
        self.last = self.current
        return self.times[-1]

    def avg(self) -> float:

        """返回平均时间。
        
        :returns: 平均时间
        :rtype: float
        """
        
        return sum(self.times) / len(self.times)

    def sum(self) -> float:

        """返回时间总和。
        
        :returns: 时间总和。
        :rtype: float
        """
        
        return sum(self.times)