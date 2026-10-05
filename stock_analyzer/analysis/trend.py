"""Trendstruktur (HH/HL/LH/LL), Ereignisse (Breakout, Breakdown, Pullback, Rebound, Konsolidierung, Trendwechsel)."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .swings import find_pivots


@dataclass
class TrendInfo:
    direction: str = "seitwärts"          # "aufwärts" | "abwärts" | "seitwärts"
    strength: float = 0.0                 # 0..1
    structure: str = ""                   # z. B. "höhere Hochs + höhere Tiefs"
    events: list = field(default_factory=list)   # [(typ, text, bias)]
    last_pivot_high: float | None = None
    last_pivot_low: float | None = None


def analyze_trend(df: pd.DataFrame, ind: pd.DataFrame, rel_vol: pd.Series, n: int = 5) -> TrendInfo:
    t = TrendInfo()
    highs, lows = find_pivots(df, n)
    close = df["Close"]
    c = float(close.iloc[-1])
    if len(highs) < 2 or len(lows) < 2:
        t.structure = "Daten nicht ausreichend für Trendstruktur"
        return t
    hh, hl = highs[-1][1] > highs[-2][1], lows[-1][1] > lows[-2][1]
    lh, ll = highs[-1][1] < highs[-2][1], lows[-1][1] < lows[-2][1]
    t.last_pivot_high, t.last_pivot_low = highs[-1][1], lows[-1][1]
    if hh and hl:
        t.direction, t.structure, base = "aufwärts", "höhere Hochs + höhere Tiefs", 0.6
    elif lh and ll:
        t.direction, t.structure, base = "abwärts", "tiefere Hochs + tiefere Tiefs", 0.6
    else:
        t.direction, base = "seitwärts", 0.2
        t.structure = "Hochs/Tiefs ohne klare Richtung (" + ("höhere Hochs" if hh else "tiefere Hochs") + ", " + ("höhere Tiefs" if hl else "tiefere Tiefs") + ")"
    # Bestätigung/Abschwächung durch SMA50-Steigung und Preislage
    sma50 = ind.get("SMA50")
    if sma50 is not None and not np.isnan(sma50.iloc[-1]) and len(sma50.dropna()) > 10:
        slope = sma50.iloc[-1] / sma50.iloc[-10] - 1
        agrees = (slope > 0 and t.direction == "aufwärts") or (slope < 0 and t.direction == "abwärts")
        base += 0.25 if agrees else (-0.15 if t.direction != "seitwärts" else 0)
        if t.direction == "seitwärts":
            t.direction = "aufwärts" if (slope > 0.02 and c > sma50.iloc[-1]) else "abwärts" if (slope < -0.02 and c < sma50.iloc[-1]) else "seitwärts"
            if t.direction != "seitwärts":
                t.structure += f"; SMA50 {'steigt' if slope > 0 else 'fällt'}"; base = 0.3
    t.strength = float(min(max(base, 0.0), 1.0))

    rv = rel_vol.iloc[-1] if len(rel_vol) else np.nan
    strong_vol = (not np.isnan(rv)) and rv >= 1.3
    # Breakout / Breakdown: Schlusskurs jenseits des letzten bestätigten Pivots (in den letzten 3 Bars erstmals)
    prev_closes = close.iloc[-4:-1]
    if c > t.last_pivot_high and (prev_closes <= t.last_pivot_high).all():
        t.events.append(("Breakout", f"Schlusskurs über letztes Swing-Hoch ({t.last_pivot_high:.2f})" + (" mit erhöhtem Volumen" if strong_vol else " ohne Volumenbestätigung"), "bullish"))
    if c < t.last_pivot_low and (prev_closes >= t.last_pivot_low).all():
        t.events.append(("Breakdown", f"Schlusskurs unter letztes Swing-Tief ({t.last_pivot_low:.2f})" + (" mit erhöhtem Volumen" if strong_vol else " ohne Volumenbestätigung"), "bearish"))
    # Pullback / Rebound: Gegenbewegung gegen den Haupttrend, Trend noch intakt
    if t.direction == "aufwärts" and c < close.iloc[-6:].max() * 0.97 and c > t.last_pivot_low:
        t.events.append(("Pullback", "Rücksetzer im intakten Aufwärtstrend (Tiefs noch nicht unterschritten)", "neutral"))
    if t.direction == "abwärts" and c > close.iloc[-6:].min() * 1.03 and c < t.last_pivot_high:
        t.events.append(("Rebound", "Erholung im Abwärtstrend (Hochs noch nicht überwunden)", "neutral"))
    # Konsolidierung: enge 15-Bar-Spanne relativ zu ATR
    if len(df) >= 15:
        span = df["High"].iloc[-15:].max() - df["Low"].iloc[-15:].min()
        a = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        if a and span < 3.5 * a:
            t.events.append(("Konsolidierung", "Kurs bewegt sich in enger Spanne (Volatilität komprimiert)", "neutral"))
    # Mögliche Trendwende: Abwärtstrend, aber letztes Tief höher als vorheriges + Kurs über EMA20 (und umgekehrt)
    ema20 = ind.get("EMA20")
    if ema20 is not None and not np.isnan(ema20.iloc[-1]):
        if t.direction == "abwärts" and hl and c > ema20.iloc[-1]:
            t.events.append(("Trendwechsel?", "Abwärtstrend wird möglicherweise verlassen: höheres Tief und Kurs über EMA20", "bullish"))
        if t.direction == "aufwärts" and ll and c < ema20.iloc[-1]:
            t.events.append(("Trendwechsel?", "Aufwärtstrend gerät ins Wanken: tieferes Tief und Kurs unter EMA20", "bearish"))
    return t
