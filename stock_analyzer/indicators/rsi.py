"""RSI nach Welles Wilder.

Designentscheidung: Wilders Glättung wird exakt umgesetzt (erster Durchschnitt = einfacher
Mittelwert der ersten `period` Veränderungen, danach rekursiv (prev*(n-1)+x)/n). Viele
Bibliotheken weichen hier leicht ab; für die Reproduzierbarkeit gegen TA-Lib/StockCharts
ist die exakte Variante wichtig.
"""
import numpy as np
import pandas as pd

# Zonen laut Anforderung. Sie sind KEIN Kauf-/Verkaufssignal, nur Einordnung.
ZONES = [(30, "überverkauft"), (40, "schwach"), (60, "neutral"), (70, "stark")]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= period:
        return pd.Series(out, index=close.index, name=f"RSI{period}")
    d = np.diff(c)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(c)):
        if i > period:
            ag = (ag * (period - 1) + gain[i - 1]) / period
            al = (al * (period - 1) + loss[i - 1]) / period
        out[i] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100 / (1 + ag / al))
    return pd.Series(out, index=close.index, name=f"RSI{period}")


def rsi_zone(value: float) -> str:
    for limit, name in ZONES:
        if value < limit:
            return name
    return "überkauft"
