"""SMA/EMA-Familie, ATR und Cross-Erkennung."""
import pandas as pd
from .macd import ema


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def add_moving_averages(df: pd.DataFrame, sma_periods=(20, 50, 200), ema_periods=(20, 50)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for n in sma_periods:
        out[f"SMA{n}"] = sma(df["Close"], n)
    for n in ema_periods:
        out[f"EMA{n}"] = ema(df["Close"], n).where(df["Close"].expanding().count() >= n)
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder-Glättung) – Maß für Volatilität, Basis für Zonenbreiten/Stops."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def crossed_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def recent(flag: pd.Series, bars: int) -> bool:
    """War das Ereignis innerhalb der letzten `bars` Bars (inkl. aktueller)?"""
    return bool(flag.iloc[-bars:].any())
