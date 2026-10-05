"""Bewertung der Einzelkomponenten (jeweils Score -100..+100 + Begründungen).

Kernidee dieses Moduls: KONTEXT statt Schwellenwert-Regeln. Beispiele:
  * RSI < 30 ist im Abwärtstrend zunächst nur Verkaufsdruck; erst Wiederanstieg, Divergenz,
    MACD-Wende usw. machen daraus ein Erholungssignal.
  * Eine Candlestick-Formation zählt viel an einer Unterstützung mit RSI/MACD-Bestätigung und wenig
    mitten im Trend ohne Bestätigung.
Jeder Score wird nur aus tatsächlich berechneten Werten abgeleitet; fehlen Daten, ist die Komponente
`available=False` und wird aus der Gewichtung herausgenommen (keine Schätzung).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ..indicators.rsi import rsi_zone
from ..indicators.moving_averages import crossed_up, crossed_down, recent
from ..indicators.volume import obv
from .support_resistance import nearest_zones


def clip(x, lo=-100.0, hi=100.0): return float(max(lo, min(hi, x)))


def label(score: float) -> str:
    for lim, name in [(-65, "Sehr stark bearish"), (-35, "Bearish"), (-12, "Leicht bearish"),
                      (12, "Neutral"), (35, "Leicht bullish"), (65, "Bullish")]:
        if score < lim:
            return name
    return "Sehr stark bullish"


@dataclass
class Component:
    name: str
    score: float = 0.0
    reasons: list = field(default_factory=list)
    available: bool = True
    details: dict = field(default_factory=dict)


def _na(name, why): return Component(name, 0.0, [why], False)


def score_rsi(ctx) -> Component:
    r = ctx.rsi.dropna()
    if len(r) < 12:
        return _na("RSI", "RSI: Daten nicht verfügbar (zu wenig Historie)")
    v, prev, slope3 = float(r.iloc[-1]), float(r.iloc[-2]), float(r.iloc[-1] - r.iloc[-4])
    td = ctx.trend.direction
    c = Component("RSI")
    direction = "steigend" if slope3 > 1.5 else "fallend" if slope3 < -1.5 else "seitwärts"
    c.details.update(value=v, zone=rsi_zone(v), direction=direction)
    s = np.clip((v - 50) / 20, -1, 1) * 12 + np.clip(slope3 / 10, -1, 1) * 10      # Lage + Richtung
    c.reasons.append(f"RSI = {v:.0f} ({rsi_zone(v)}), {direction}")
    min10, max10 = float(r.iloc[-10:].min()), float(r.iloc[-10:].max())
    if min10 < 30:
        if v >= 30 and v > prev:
            s += 25; c.reasons.append("RSI steigt aus dem überverkauften Bereich (war zuletzt unter 30) – mögliche Erholung")
            if v >= 45: s += 5; c.reasons.append("RSI nähert sich/überschreitet 50")
        elif v < 30 and slope3 > 0:
            s += 8; c.reasons.append("RSI dreht aus sehr tiefer Zone nach oben, ist aber noch unter 30 – noch keine Bestätigung")
        elif v < 30:
            s -= 5
            c.reasons.append("RSI ist überverkauft und fällt weiter" + (" – im Abwärtstrend zeigt das zunächst starken Verkaufsdruck, nicht automatisch eine Chance" if td == "abwärts" else ""))
    if max10 > 70:
        if v < 70 and v < prev:
            s -= 25; c.reasons.append("RSI fällt aus dem überkauften Bereich zurück (war zuletzt über 70) – mögliche Abkühlung")
        elif v >= 70 and slope3 < 0:
            s -= 10; c.reasons.append("RSI ist überkauft und dreht nach unten")
        elif v >= 70:
            c.reasons.append("RSI überkauft" + (", kann im starken Aufwärtstrend aber länger so bleiben" if td == "aufwärts" else " – erhöhtes Rückschlagrisiko"))
    if recent(crossed_up(r, pd.Series(50.0, index=r.index)), 3):
        s += 15; c.reasons.append("RSI hat die 50er-Marke nach oben durchbrochen (Momentum-Wechsel zugunsten der Käufer)")
    if recent(crossed_down(r, pd.Series(50.0, index=r.index)), 3):
        s -= 15; c.reasons.append("RSI ist unter die 50er-Marke gefallen (Momentum-Wechsel zugunsten der Verkäufer)")
    for d in ctx.divs:
        if d.indicator == "RSI":
            s += 35 if d.kind == "bullish" else -35
            c.reasons.insert(1, f"{'Bullishe' if d.kind == 'bullish' else 'Bearishe'} RSI-Divergenz: Kurs {d.price1:.2f} → {d.price2:.2f}, RSI {d.ind1:.0f} → {d.ind2:.0f}")
    c.score = clip(s); return c


def score_macd(ctx) -> Component:
    m = ctx.macd.dropna()
    if len(m) < 6:
        return _na("MACD", "MACD: Daten nicht verfügbar (zu wenig Historie)")
    line, sig, hist = m["macd"], m["signal"], m["hist"]
    mv, h0, h1, h2 = float(line.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2]), float(hist.iloc[-3])
    c = Component("MACD"); s = 0.0
    c.details.update(macd=mv, signal=float(sig.iloc[-1]), hist=h0)
    s += 10 if mv > sig.iloc[-1] else -10
    s += 10 if mv > 0 else -10
    c.reasons.append(f"MACD {'über' if mv > 0 else 'unter'} der Nulllinie und {'über' if mv > sig.iloc[-1] else 'unter'} der Signallinie")
    up, dn = crossed_up(line, sig), crossed_down(line, sig)
    if recent(up, 3):
        s += 35 if mv > 0 else 22
        c.reasons.insert(0, "Bullisher MACD-Crossover" + (" oberhalb der Nulllinie (Trendfortsetzung, stärkeres Signal)" if mv > 0 else " unterhalb der Nulllinie (frühes Wendesignal, schwächer als über Null)"))
    if recent(dn, 3):
        s -= 35 if mv < 0 else 22
        c.reasons.insert(0, "Bearisher MACD-Crossover" + (" unterhalb der Nulllinie (Trendfortsetzung nach unten)" if mv < 0 else " oberhalb der Nulllinie (frühe Schwäche, schwächer als unter Null)"))
    if recent(crossed_up(line, pd.Series(0.0, index=line.index)), 3):
        s += 15; c.reasons.append("MACD hat die Nulllinie nach oben überschritten")
    if recent(crossed_down(line, pd.Series(0.0, index=line.index)), 3):
        s -= 15; c.reasons.append("MACD hat die Nulllinie nach unten unterschritten")
    rising = h0 > h1 > h2
    falling = h0 < h1 < h2
    if h0 < 0 and h0 > h1:
        s += 10 if rising else 6
        c.reasons.append("Histogramm noch negativ, aber die Balken werden kleiner: negatives Momentum nimmt ab (noch kein Kaufsignal, mögliche Erholung)")
    elif h0 > 0 and h0 > h1:
        s += 15; c.reasons.append("Histogramm positiv und wächst: Aufwärtsmomentum nimmt zu")
    elif h0 > 0 and h0 < h1:
        s -= 10; c.reasons.append("Histogramm noch positiv, aber schrumpft: Aufwärtsmomentum lässt nach")
    elif h0 < 0 and h0 < h1:
        s -= 15; c.reasons.append("Histogramm negativ und wird größer: Abwärtsmomentum nimmt zu")
    c.details["momentum"] = "zunehmend" if (h0 > h1) else "abnehmend"
    for d in ctx.divs:
        if d.indicator == "MACD":
            s += 30 if d.kind == "bullish" else -30
            c.reasons.insert(1, f"{'Bullishe' if d.kind == 'bullish' else 'Bearishe'} MACD-Divergenz (Kurs {d.price1:.2f} → {d.price2:.2f})")
    c.score = clip(s); return c


def score_candles(ctx) -> Component:
    c = Component("Candlesticks"); n = len(ctx.df)
    sup, res, inside, _, _ = nearest_zones(ctx.zones, float(ctx.df["Close"].iloc[-1]))
    price, atr = float(ctx.df["Close"].iloc[-1]), ctx.atr
    rsi_s, hist = ctx.rsi.dropna(), ctx.macd["hist"].dropna()
    rsi_up = len(rsi_s) > 3 and rsi_s.iloc[-1] > rsi_s.iloc[-3]
    hist_up = len(hist) > 2 and hist.iloc[-1] > hist.iloc[-2]
    s, shown = 0.0, []
    for p in [p for p in ctx.patterns if p.pos >= n - 3]:
        age = n - 1 - p.pos
        decay = [1.0, 0.7, 0.5][age]
        sign = {"bullish": 1, "bearish": -1, "neutral": 0}[p.bias]
        near_sup = sup is not None and price - sup.high <= atr or (inside is not None and price <= inside.mid)
        near_res = res is not None and res.low - price <= atr or (inside is not None and price >= inside.mid)
        if sign > 0:
            conf = [x for x, ok in (("RSI dreht nach oben", rsi_up and rsi_s.iloc[-1] < 55), ("MACD-Histogramm verbessert sich", hist_up)) if ok]
            at_level = near_sup
        elif sign < 0:
            conf = [x for x, ok in (("RSI dreht nach unten", (not rsi_up) and rsi_s.iloc[-1] > 45), ("MACD-Histogramm verschlechtert sich", not hist_up)) if ok]
            at_level = near_res
        else:
            conf, at_level = [], False
        mult = 1.0 + (0.5 if at_level else 0) + 0.25 * len(conf)
        against = (sign > 0 and ctx.trend.direction == "abwärts" and ctx.trend.strength >= 0.5) or \
                  (sign < 0 and ctx.trend.direction == "aufwärts" and ctx.trend.strength >= 0.5)
        if against and not conf and not at_level:
            mult *= 0.4
        s += sign * p.strength * 40 * mult * decay
        txt = f"{p.name} ({p.bias}), " + ("an " + ("Unterstützung" if sign > 0 else "Widerstand") if at_level else "ohne Bezug zu einer Kurszone")
        txt += ("; bestätigt durch: " + ", ".join(conf)) if conf else ("; nicht durch RSI/MACD bestätigt" if sign else "")
        if against and not conf and not at_level:
            txt += " – steht gegen den Haupttrend, daher nur schwach gewertet"
        shown.append(dict(name=p.name, time=p.time, bias=p.bias, strength=p.strength, at_level=bool(at_level),
                          confirmed=bool(conf), confirmations=conf, text=txt))
        c.reasons.append(txt)
    c.details["patterns"] = shown
    if not shown:
        c.reasons.append("Keine Candlestick-Formation in den letzten 3 Kerzen")
    c.score = clip(s); return c


def score_trend(ctx) -> Component:
    t = ctx.trend; c = Component("Trend")
    sign = {"aufwärts": 1, "abwärts": -1, "seitwärts": 0}[t.direction]
    s = sign * t.strength * 60
    c.reasons.append(f"{t.direction.capitalize()}strend ({t.structure})" if t.direction != "seitwärts" else f"Seitwärtsbewegung ({t.structure})")
    for typ, txt, bias in t.events:
        w = {"Breakout": 25, "Breakdown": -25, "Trendwechsel?": 15, "Pullback": 5, "Rebound": -5}.get(typ, 0)
        if typ == "Trendwechsel?" and bias == "bearish": w = -15
        s += w; c.reasons.append(f"{typ}: {txt}")
    c.details["direction"] = t.direction
    c.score = clip(s); return c


def score_sr(ctx) -> Component:
    price, atr = float(ctx.df["Close"].iloc[-1]), ctx.atr
    sup, res, inside, _, _ = nearest_zones(ctx.zones, price)
    c = Component("Unterstützung/Widerstand"); s = 0.0
    if sup is None and res is None and inside is None:
        return _na("Unterstützung/Widerstand", "Keine belastbaren Zonen erkannt – Daten nicht ausreichend")
    if inside is not None:
        s += 15 if price >= inside.mid else -15
        c.reasons.append(f"Kurs liegt in der Zone {inside.low:.2f}–{inside.high:.2f} ({', '.join(inside.sources)}) und {'hält sie bisher als Unterstützung' if price >= inside.mid else 'stößt von unten an sie als Widerstand'}")
    d_sup = (price - sup.high) / atr if sup else None
    d_res = (res.low - price) / atr if res else None
    if sup: c.reasons.append(f"Unterstützungszone {sup.low:.2f}–{sup.high:.2f} (Abstand {d_sup:.1f} ATR, {sup.touches} Berührungen)")
    if res: c.reasons.append(f"Widerstandszone {res.low:.2f}–{res.high:.2f} (Abstand {d_res:.1f} ATR, {res.touches} Berührungen)")
    if sup and res:
        s += np.clip((d_res - d_sup) / (d_res + d_sup + 1e-9), -1, 1) * 45
    elif sup:
        s += 15; c.reasons.append("Kein Widerstand in Reichweite erkannt")
    elif res:
        s -= 15; c.reasons.append("Keine Unterstützung in Reichweite erkannt")
    c.details.update(d_sup=d_sup, d_res=d_res)
    c.score = clip(s); return c


def score_volume(ctx) -> Component:
    df = ctx.df
    if df["Volume"].iloc[-30:].sum() <= 0 or np.isnan(ctx.rel_vol.iloc[-1]):
        return _na("Volumen", "Volumen: Daten nicht verfügbar")
    rv = float(ctx.rel_vol.iloc[-1]); c = Component("Volumen"); s = 0.0
    bar_dir = np.sign(df["Close"].iloc[-1] - df["Open"].iloc[-1])
    c.details["rel_volume"] = rv
    if rv >= 1.3:
        s += bar_dir * min((rv - 1) * 25, 40)
        c.reasons.append(f"Überdurchschnittliches Volumen ({rv:.1f}× Ø) bei {'steigendem' if bar_dir > 0 else 'fallendem'} Kurs")
    elif rv < 0.7:
        c.reasons.append(f"Unterdurchschnittliches Volumen ({rv:.1f}× Ø) – Bewegungen sind weniger verlässlich")
    else:
        c.reasons.append(f"Volumen im Normalbereich ({rv:.1f}× Ø)")
    o = obv(df)
    if len(o) > 12:
        up = o.iloc[-1] > o.iloc[-11]; s += 10 if up else -10
        c.reasons.append("OBV steigt (Zufluss überwiegt)" if up else "OBV fällt (Abfluss überwiegt)")
    for typ, _, _ in ctx.trend.events:
        if typ == "Breakout":
            s += 25 if rv >= 1.3 else (-10 if rv < 0.8 else 0)
            c.reasons.append("Ausbruch mit erhöhtem Volumen bestätigt" if rv >= 1.3 else "Ausbruch ohne Volumenbestätigung – Gefahr eines Fehlausbruchs" if rv < 0.8 else "Ausbruch mit durchschnittlichem Volumen")
        if typ == "Breakdown":
            s -= 25 if rv >= 1.3 else 0
            c.reasons.append("Bruch mit erhöhtem Volumen – Verkaufsdruck bestätigt" if rv >= 1.3 else "Bruch ohne besondere Volumenbestätigung")
    c.score = clip(s); return c


def score_ma(ctx) -> Component:
    ind, price = ctx.ma, float(ctx.df["Close"].iloc[-1]); c = Component("Gleitende Durchschnitte"); s = 0.0
    cols = [k for k in ind.columns if not np.isnan(ind[k].iloc[-1])]
    if not cols:
        return _na("Gleitende Durchschnitte", "Gleitende Durchschnitte: Daten nicht verfügbar")
    for col in [k for k in cols if k.startswith("SMA")]:
        n = int(col[3:]); w = 12 if n >= 150 else 8
        above = price > ind[col].iloc[-1]; s += w if above else -w
        c.reasons.append(f"Kurs {'über' if above else 'unter'} {col} ({ind[col].iloc[-1]:.2f}, Abstand {(price / ind[col].iloc[-1] - 1) * 100:+.1f} %)")
    sm = sorted([k for k in cols if k.startswith("SMA")], key=lambda k: int(k[3:]))
    if len(sm) >= 2:
        a, b = ind[sm[-2]], ind[sm[-1]]               # z. B. SMA50 vs SMA200
        s += 8 if a.iloc[-1] > b.iloc[-1] else -8
        if recent(crossed_up(a, b), 10): s += 15; c.reasons.append(f"Golden Cross ({sm[-2]} über {sm[-1]}) in den letzten 10 Bars")
        elif recent(crossed_down(a, b), 10): s -= 15; c.reasons.append(f"Death Cross ({sm[-2]} unter {sm[-1]}) in den letzten 10 Bars")
        else: c.reasons.append(f"{sm[-2]} liegt {'über' if a.iloc[-1] > b.iloc[-1] else 'unter'} {sm[-1]}")
    if "EMA20" in cols and "EMA50" in cols:
        e1, e2 = ind["EMA20"], ind["EMA50"]; s += 8 if e1.iloc[-1] > e2.iloc[-1] else -8
        if recent(crossed_up(e1, e2), 5): s += 10; c.reasons.append("Bullisher EMA20/EMA50-Crossover")
        if recent(crossed_down(e1, e2), 5): s -= 10; c.reasons.append("Bearisher EMA20/EMA50-Crossover")
    if "SMA50" in cols:
        dist = price / ind["SMA50"].iloc[-1] - 1
        if dist > 0.12: s -= 8; c.reasons.append(f"Kurs weit über SMA50 ({dist * 100:.0f} %) – überdehnt, Rücksetzer-Risiko")
        elif dist < -0.12: c.reasons.append(f"Kurs weit unter SMA50 ({dist * 100:.0f} %) – großer Abstand zum Mittelwert")
    c.score = clip(s); return c


SCORERS = [score_rsi, score_macd, score_candles, score_trend, score_sr, score_volume, score_ma]
