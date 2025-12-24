"""
日志记录工具
"""

import logging
import sys
from typing import Optional


class Logger:
    """
    简单的日志记录器
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # 设置日志级别
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误"""
        self.logger.critical(message)


def get_logger(name: str, level: str = "INFO") -> Logger:
    """获取日志记录器的便捷函数"""
    return Logger(name, level)
