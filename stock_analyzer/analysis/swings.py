"""Swing-Hochs/-Tiefs (Pivots) und Divergenz-Erkennung.

Look-Ahead-Schutz: Ein Pivot mit Fenster n braucht n Folgebars zur Bestätigung. Deshalb werden
Pivots nur bis Position len-1-n zurückgegeben – die Erkennung kennt nichts, was erst später sichtbar wäre.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def find_pivots(df: pd.DataFrame, n: int = 5):
    """Gibt (pivot_highs, pivot_lows) als Listen von (pos, preis) zurück – nur bestätigte Pivots."""
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max() and h[i] > h[i - n:i].max():
            highs.append((i, h[i]))
        if l[i] == l[i - n:i + n + 1].min() and l[i] < l[i - n:i].min():
            lows.append((i, l[i]))
    return highs, lows


@dataclass
class Divergence:
    kind: str           # "bullish" | "bearish"
    indicator: str      # "RSI" | "MACD"
    pos1: int
    pos2: int
    time1: object
    time2: object
    price1: float
    price2: float
    ind1: float
    ind2: float


def find_divergences(df: pd.DataFrame, ind: pd.Series, name: str, n: int = 5, max_age: int = 30):
    """Bullish: Kurs tieferes Tief, Indikator höheres Tief. Bearish: Kurs höheres Hoch, Indikator niedrigeres Hoch.
    Nur die letzten zwei Pivots, deren zweites höchstens `max_age` Bars alt ist."""
    highs, lows = find_pivots(df, n)
    last = len(df) - 1
    out = []
    if len(lows) >= 2 and last - lows[-1][0] <= max_age:
        (p1, v1), (p2, v2) = lows[-2], lows[-1]
        i1, i2 = ind.iloc[p1], ind.iloc[p2]
        if p2 - p1 >= 5 and v2 < v1 and i2 > i1 and not (np.isnan(i1) or np.isnan(i2)):
            out.append(Divergence("bullish", name, p1, p2, df.index[p1], df.index[p2], v1, v2, float(i1), float(i2)))
    if len(highs) >= 2 and last - highs[-1][0] <= max_age:
        (p1, v1), (p2, v2) = highs[-2], highs[-1]
        i1, i2 = ind.iloc[p1], ind.iloc[p2]
        if p2 - p1 >= 5 and v2 > v1 and i2 < i1 and not (np.isnan(i1) or np.isnan(i2)):
            out.append(Divergence("bearish", name, p1, p2, df.index[p1], df.index[p2], v1, v2, float(i1), float(i2)))
    return out
