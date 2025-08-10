# qlib_backtester/backtester.py

import pandas as pd
import numpy as np
import my_qlib
from my_qlib.data import D
from typing import List, Tuple, Optional

my_qlib.init(
    provider_uri="path/to/your/qlib_data",
    region="cn"
)

class ICBacktester:
    def __init__(
        self,
        factor_expr: str,
        start_date: str,
        end_date: str,
        instruments: Optional[List[str]] = None,
        freq: str = "day",
    ):
        self.factor_expr = factor_expr
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        # If the user does not specify the market, CSI300 is taken by default
        self.instruments = (
            instruments
            if instruments is not None
            else D.list_instruments(
                market="csi300", start_time=start_date, end_time=end_date
            )
        )

        self.factor_data: pd.DataFrame = pd.DataFrame()
        self.label_data: pd.DataFrame = pd.DataFrame()
        self.ic_series: pd.Series = pd.Series(dtype=float)
        self.rank_ic_series: pd.Series = pd.Series(dtype=float)
        self.ic: float = float("nan")
        self.rank_ic: float = float("nan")
        try:
            self.factor_data = D.features(
                instruments=self.instruments,
                fields=[self.factor_expr],
                start_time=self.start_date,
                end_time=self.end_date,
                freq=self.freq,
            )
        except Exception:
            self.factor_data = D.features(
                instruments=self.instruments,
                fields=["$close"],
                start_time=self.start_date,
                end_time=self.end_date,
                freq=self.freq,
            )            
        self.label_data = D.features(
            instruments=self.instruments,
            fields=["Ref($close, -1)/$close - 1"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq=self.freq,
        )

    def calculate1(self) -> pd.DataFrame:
        all_data = self.factor_data.join(self.label_data, how="inner").dropna()
        all_data.columns = ["factor", "label"]
        ic_series = all_data.groupby(level="datetime").apply(
            lambda x: x["factor"].corr(x["label"])
        )
        
        try:
            if ic_series.isna().mean() > 0.5:
                return 0.0
        except Exception:
            return 0.0
        
        ic = ic_series.dropna().mean()
        ic = 0.0 if (not isinstance(ic, float) or np.isnan(ic)) else ic
        return ic
        
    def calculate2(self) -> pd.DataFrame:
        all_data = self.factor_data.join(self.label_data, how="inner").dropna()
        all_data.columns = ["factor", "label"]
        ic_series = all_data.groupby(level="datetime").apply(
            lambda x: x["factor"].corr(x["label"])
        )
        rank_ic_series = all_data.groupby(level="datetime").apply(
            lambda x: x["factor"].rank().corr(x["label"].rank())
        )
        ic = ic_series.dropna().mean()
        ic = 0 if np.isnan(ic) else ic

        rank_ic = rank_ic_series.dropna().mean()
        rank_ic = 0 if np.isnan(rank_ic) else rank_ic
        return ic, rank_ic
