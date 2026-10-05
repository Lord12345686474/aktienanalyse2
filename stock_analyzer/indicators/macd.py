"""MACD (EMA fast - EMA slow), Signallinie (EMA des MACD) und Histogramm.

EMA: alpha = 2/(n+1), rekursiv, Start mit dem ersten Wert (adjust=False) – so rechnen
gängige Charting-Plattformen; nach einigen Dutzend Bars ist der Startwert irrelevant.
"""
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    out = pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})
    out.iloc[: slow - 1] = float("nan")          # Einschwingphase nicht verwenden
    return out
