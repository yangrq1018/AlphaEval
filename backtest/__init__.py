# qlib_backtester/__init__.py

"""
qlib_backtester
===============

一个基于 Qlib 的因子回测包，直接输入因子表达式字符串和回测区间，
即可快速计算全样本的每日 IC 和 Rank-IC 均值，以及时间序列。
"""

__version__ = "0.1.0"

from .backtester import FactorBacktester

__all__ = ["FactorBacktester"]
