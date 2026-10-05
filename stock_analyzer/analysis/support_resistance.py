"""Unterstützungs-/Widerstandszonen (keine Einzelkurse).

Kandidaten: Swing-Hochs/-Tiefs, Volumenprofil-Knoten, SMA50/200, runde Marken. Nahe Kandidaten werden zu einer
Zone geclustert (Toleranz ~ 0.6 ATR); Stärke = Anzahl Berührungen + Konfluenz mit MAs/Volumen/runden Marken.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .swings import find_pivots
from ..indicators.volume import volume_profile_levels


@dataclass
class Zone:
    low: float
    high: float
    strength: float          # gewichtete Berührungen
    touches: int
    sources: list

    @property
    def mid(self): return (self.low + self.high) / 2


def _round_step(price: float) -> float:
    return 1 if price < 20 else 5 if price < 100 else 10 if price < 500 else 50 if price < 2000 else 100


def find_zones(df: pd.DataFrame, ind: pd.DataFrame, atr_val: float, n: int = 5, lookback: int = 300):
    d = df.iloc[-lookback:]
    off = len(df) - len(d)
    price = float(df["Close"].iloc[-1])
    cands = []                                               # (preis, gewicht, quelle)
    highs, lows = find_pivots(d, n)
    cands += [(p, 1.0, "Swing-Hoch") for _, p in highs] + [(p, 1.0, "Swing-Tief") for _, p in lows]
    cands += [(p, 1.5, "Volumenknoten") for p in volume_profile_levels(d)]
    for col in ("SMA50", "SMA200"):
        if col in ind and not np.isnan(ind[col].iloc[-1]):
            cands.append((float(ind[col].iloc[-1]), 1.5, col))
    step = _round_step(price)
    for k in range(-6, 7):                                   # runde Marken nahe dem Kurs
        lvl = round(price / step) * step + k * step
        if abs(lvl - price) <= 8 * atr_val and lvl > 0:
            cands.append((lvl, 0.5, "runde Marke"))
    if not cands:
        return []
    tol = max(0.6 * atr_val, price * 0.004)
    cands.sort()
    clusters, cur = [], [cands[0]]
    for cnd in cands[1:]:
        if cnd[0] - np.mean([x[0] for x in cur]) <= tol:
            cur.append(cnd)
        else:
            clusters.append(cur); cur = [cnd]
    clusters.append(cur)
    zones = []
    for cl in clusters:
        prices = [x[0] for x in cl]
        touches = sum(1 for x in cl if x[2] in ("Swing-Hoch", "Swing-Tief"))
        strength = sum(x[1] for x in cl)
        if strength < 2.0:                                    # einzelner schwacher Kandidat -> keine Zone
            continue
        half = max((max(prices) - min(prices)) / 2, 0.25 * atr_val)
        mid = float(np.mean(prices))
        zones.append(Zone(mid - half, mid + half, strength, touches, sorted({x[2] for x in cl})))
    return zones


def nearest_zones(zones, price: float):
    """(Unterstützung unterhalb, Widerstand oberhalb, ggf. Zone in der der Kurs liegt) – jeweils nach Nähe."""
    inside = next((z for z in zones if z.low <= price <= z.high), None)
    below = sorted([z for z in zones if z.high < price], key=lambda z: price - z.high)
    above = sorted([z for z in zones if z.low > price], key=lambda z: z.low - price)
    return (below[0] if below else None, above[0] if above else None, inside, below, above)
