"""Synthetische OHLCV-Daten nur für TESTS (nie für die Anzeige echter Analysen)."""
import numpy as np, pandas as pd


def make_df(n=600, seed=1, drift=0.0004, vol=0.015, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": rng.integers(8e5, 2e6, n).astype(float)}, index=idx)
