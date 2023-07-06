# coding:utf-8
#
# pybind11_ke/utils/Timer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on July 6, 2023
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

        self.times = []
        self.restart()

    def restart(self):

        """重启计时器。"""
        
        self.current = self.last = time.time()

    def stop(self):

        """停止计时器并将时间记录在列表中。
        
        :returns: 返回最后一次的间隔时间。
        :rtype: float
        """
        
        self.current = time.time()
        self.times.append(self.current - self.last)
        self.last = self.current
        return self.times[-1]

    def avg(self):

        """返回平均时间。
        
        :returns: 平均时间
        :rtype: float
        """
        
        return sum(self.times) / len(self.times)

    def sum(self):

        """返回时间总和。
        
        :returns: 时间总和。
        :rtype: float
        """
        
        return sum(self.times)