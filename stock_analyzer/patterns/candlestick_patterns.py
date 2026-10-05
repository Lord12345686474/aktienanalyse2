"""Candlestick-Erkennung (kausal: Bar i nutzt nur Bars <= i).

Jede Formation liefert Name, Bias und eine Basisstärke (0..1). Der KONTEXT (vorheriger Trend,
Unterstützung, RSI/MACD) wird erst im Scoring bewertet – hier wird nur die Form erkannt.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Pattern:
    pos: int            # Bar-Position im DataFrame
    time: object        # Zeitstempel der Bar
    name: str
    bias: str           # "bullish" | "bearish" | "neutral"
    strength: float     # Basisstärke 0..1 (nur Form)


def _prior_trend(close: np.ndarray, i: int, look: int = 6, thr: float = 0.02) -> int:
    """-1 Abwärts, +1 Aufwärts, 0 unklar – gemessen VOR der Formation (Bars i-look .. i-1)."""
    if i - look < 0:
        return 0
    ret = close[i - 1] / close[i - look] - 1
    return -1 if ret < -thr else (1 if ret > thr else 0)


def detect_patterns(df: pd.DataFrame) -> list[Pattern]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    body, rng = np.abs(c - o), h - l
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    avg_body = pd.Series(body).rolling(14, min_periods=3).mean().shift(1).to_numpy()
    found: list[Pattern] = []

    def add(i, name, bias, s):
        found.append(Pattern(i, df.index[i], name, bias, float(min(max(s, 0.1), 1.0))))

    for i in range(2, len(df)):
        if rng[i] <= 0:
            continue
        trend = _prior_trend(c, i)
        big = body[i] > (avg_body[i] if not np.isnan(avg_body[i]) else body[i])
        # --- Ein-Kerzen-Formationen ---
        if body[i] <= 0.1 * rng[i]:
            add(i, "Doji", "neutral", 0.4)
        elif body[i] <= 0.3 * rng[i] and upper[i] >= body[i] and lower[i] >= body[i]:
            add(i, "Spinning Top", "neutral", 0.3)
        hammer_shape = body[i] > 0 and lower[i] >= 2 * body[i] and upper[i] <= max(0.15 * rng[i], 0.5 * body[i])
        inv_shape = body[i] > 0 and upper[i] >= 2 * body[i] and lower[i] <= max(0.15 * rng[i], 0.5 * body[i])
        if hammer_shape and trend == -1:
            add(i, "Hammer", "bullish", 0.6 + 0.2 * (c[i] > o[i]))
        elif hammer_shape and trend == 1:
            add(i, "Hanging Man", "bearish", 0.5 + 0.1 * (c[i] < o[i]))
        if inv_shape and trend == -1:
            add(i, "Inverted Hammer", "bullish", 0.45)     # braucht Bestätigung durch Folgekerze
        elif inv_shape and trend == 1:
            add(i, "Shooting Star", "bearish", 0.65 + 0.15 * (c[i] < o[i]))
        # --- Zwei-Kerzen-Formationen ---
        pb, pr = body[i - 1], rng[i - 1]
        prev_red, prev_green = c[i - 1] < o[i - 1], c[i - 1] > o[i - 1]
        if prev_red and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] > pb and trend == -1:
            add(i, "Bullish Engulfing", "bullish", 0.7 + 0.2 * big)
        if prev_green and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] > pb and trend == 1:
            add(i, "Bearish Engulfing", "bearish", 0.7 + 0.2 * big)
        mid_prev = (o[i - 1] + c[i - 1]) / 2
        if prev_red and pb > 0.5 * pr and c[i] > o[i] and o[i] < c[i - 1] and mid_prev < c[i] < o[i - 1] and trend == -1:
            add(i, "Piercing Pattern", "bullish", 0.6)
        if prev_green and pb > 0.5 * pr and c[i] < o[i] and o[i] > c[i - 1] and o[i - 1] > c[i] > mid_prev and trend == 1:
            add(i, "Dark Cloud Cover", "bearish", 0.6)
        # --- Drei-Kerzen-Formationen (Stern) ---
        b1, b2 = body[i - 2], body[i - 1]
        t3 = _prior_trend(c, i - 1)
        mid1 = (o[i - 2] + c[i - 2]) / 2
        if c[i - 2] < o[i - 2] and b1 > 0.5 * rng[i - 2] and b2 <= 0.35 * b1 and c[i] > o[i] and c[i] > mid1 and t3 == -1:
            add(i, "Morning Star", "bullish", 0.85)
        if c[i - 2] > o[i - 2] and b1 > 0.5 * rng[i - 2] and b2 <= 0.35 * b1 and c[i] < o[i] and c[i] < mid1 and t3 == 1:
            add(i, "Evening Star", "bearish", 0.85)
    return found
