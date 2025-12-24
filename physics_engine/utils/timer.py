"""
性能计时器工具
用于测量代码执行时间
"""

import time
from typing import Dict, Optional


class Timer:
    """
    性能计时器，支持嵌套计时
    """
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.total_times: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = {}
    
    def start(self, name: str):
        """开始计时"""
        self.start_times[name] = time.perf_counter()
    
    def end(self, name: str) -> float:
        """
        结束计时并返回经过的时间
        
        Returns:
            经过的时间（秒）
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.perf_counter() - self.start_times[name]
        
        # 更新统计信息
        if name not in self.total_times:
            self.total_times[name] = 0.0
            self.call_counts[name] = 0
        
        self.total_times[name] += elapsed
        self.call_counts[name] += 1
        
        del self.start_times[name]
        return elapsed
    
    def get_average(self, name: str) -> Optional[float]:
        """获取平均时间"""
        if name not in self.total_times or self.call_counts[name] == 0:
            return None
        return self.total_times[name] / self.call_counts[name]
    
    def get_total(self, name: str) -> float:
        """获取总时间"""
        return self.total_times.get(name, 0.0)
    
    def get_count(self, name: str) -> int:
        """获取调用次数"""
        return self.call_counts.get(name, 0)
    
    def reset(self, name: str = None):
        """重置计时器"""
        if name is None:
            self.start_times.clear()
            self.total_times.clear()
            self.call_counts.clear()
        else:
            self.start_times.pop(name, None)
            self.total_times.pop(name, None)
            self.call_counts.pop(name, None)
    
    def report(self) -> str:
        """生成计时报告"""
        if not self.total_times:
            return "No timing data available"
        
        lines = ["Timer Report:"]
        lines.append("-" * 50)
        lines.append(f"{'Name':<20} {'Total':<10} {'Count':<8} {'Average':<10}")
        lines.append("-" * 50)
        
        for name in sorted(self.total_times.keys()):
            total = self.total_times[name]
            count = self.call_counts[name]
            avg = total / count if count > 0 else 0.0
            lines.append(f"{name:<20} {total:<10.4f} {count:<8} {avg:<10.4f}")
        
        return "\n".join(lines)


class PerformanceProfiler:
    """
    性能分析器，使用上下文管理器语法
    """
    
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name
    
    def __enter__(self):
        self.timer.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.end(self.name)


def profile(timer: Timer, name: str) -> PerformanceProfiler:
    """
    创建性能分析器的便捷函数
    
    使用方法:
    timer = Timer()
    with profile(timer, "my_function"):
        # 要计时的代码
        pass
    """
    return PerformanceProfiler(timer, name)
