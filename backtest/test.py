from .backtester import FactorBacktester
from qlib.data import D

bt = FactorBacktester(
    factor_expr="-1 * Std(Div($close, Ref($close, 1)), 5)",
    start_date="2020-01-01",
    end_date="2024-12-31",
    instruments=D.instruments(market='all'),
    long_threshold=0.2,
    short_threshold=0.2,
)

perf = bt.run()
print(perf)