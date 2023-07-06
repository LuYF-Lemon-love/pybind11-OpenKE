import time
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.restart()

    def restart(self):
        """重启计时器"""
        self.current = self.last = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.current = time.time()
        self.times.append(self.current - self.last)
        self.last = self.current
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)