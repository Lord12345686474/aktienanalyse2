"""Signal-Engine: orchestriert alle Module für EINEN Timeframe und trifft die Entscheidung.

Ablauf: Indikatoren -> Strukturen (Pivots, Trend, Zonen, Divergenzen, Candlesticks) -> Komponenten-Scores
-> gewichteter Gesamtscore -> Konflikt-/Kontextprüfung -> Entscheidung (KAUFEN / ABWARTEN / VERKAUFEN)
-> Risiken, Einstiegs-/Stop-/Zielzonen, Erklärungstext.

Wichtige Sicherheitsregel: KAUFEN/VERKAUFEN erfordert (a) Gesamtscore über der Schwelle, (b) mindestens
`min_agreeing_components` Komponenten in gleicher Richtung, (c) RSI und MACD dürfen nicht widersprechen,
(d) gegen einen intakten Haupttrend nur mit bestätigtem Wendesignal. Sonst: ABWARTEN.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ..config import Settings
from ..indicators.rsi import rsi as calc_rsi
from ..indicators.macd import macd as calc_macd
from ..indicators.moving_averages import add_moving_averages, atr as calc_atr, crossed_up, crossed_down, recent
from ..indicators.volume import relative_volume
from ..patterns.candlestick_patterns import detect_patterns
from .swings import find_divergences
from .trend import analyze_trend
from .support_resistance import find_zones, nearest_zones
from .scoring import SCORERS, label, clip

MIN_BARS = 60


@dataclass
class Ctx:
    df: pd.DataFrame; rsi: pd.Series; macd: pd.DataFrame; ma: pd.DataFrame; rel_vol: pd.Series
    atr: float; trend: object; zones: list; patterns: list; divs: list; settings: Settings


@dataclass
class Analysis:
    ok: bool
    message: str = ""
    df: pd.DataFrame | None = None
    ctx: Ctx | None = None
    components: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    total: float = 0.0
    label: str = "Neutral"
    decision: str = "ABWARTEN"
    strength: float = 0.0
    conflicts: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    explanation: str = ""
    summary_reasons: list = field(default_factory=list)

    def component(self, name):
        return next((c for c in self.components if c.name == name), None)


def build_context(df: pd.DataFrame, st: Settings) -> Ctx:
    r = calc_rsi(df["Close"], st.rsi_period)
    m = calc_macd(df["Close"], st.macd_fast, st.macd_slow, st.macd_signal)
    ma = add_moving_averages(df, st.sma_periods, st.ema_periods)
    rv = relative_volume(df, st.volume_ma)
    a = float(calc_atr(df, st.atr_period).iloc[-1])
    trend = analyze_trend(df, ma, rv, st.pivot_window)
    zones = find_zones(df, ma, a, st.pivot_window)
    divs = find_divergences(df, r, "RSI", st.pivot_window, st.divergence_max_age) + \
           find_divergences(df, m["macd"], "MACD", st.pivot_window, st.divergence_max_age)
    return Ctx(df, r, m, ma, rv, a, trend, zones, detect_patterns(df), divs, st)


def analyze(df: pd.DataFrame, st: Settings | None = None) -> Analysis:
    st = st or Settings()
    if df is None or len(df) < MIN_BARS:
        return Analysis(False, f"Daten nicht ausreichend (mindestens {MIN_BARS} Kerzen nötig, vorhanden: {0 if df is None else len(df)}).")
    ctx = build_context(df, st)
    if not np.isfinite(ctx.atr) or ctx.atr <= 0:
        return Analysis(False, "ATR nicht berechenbar – Daten nicht verfügbar.")
    comps = [f(ctx) for f in SCORERS]
    w_all = st.normalized_weights()
    avail = [c for c in comps if c.available]
    wsum = sum(w_all.get(c.name, 0) for c in avail) or 1.0
    weights = {c.name: w_all.get(c.name, 0) / wsum for c in avail}      # nicht verfügbare Komponenten: Gewicht umverteilt
    total = clip(sum(weights[c.name] * c.score for c in avail))
    res = Analysis(True, df=df, ctx=ctx, components=comps, weights=weights, total=total, label=label(total))
    _decide(res, st)
    res.risks = _risks(res)
    res.levels = _levels(res)
    res.explanation = _explain(res)
    return res


def _decide(res: Analysis, st: Settings):
    ctx, comps = res.ctx, {c.name: c for c in res.components}
    avail = [c for c in res.components if c.available]
    bulls = [c for c in avail if c.score > 15]; bears = [c for c in avail if c.score < -15]
    rs, ms = comps["RSI"].score, comps["MACD"].score
    # --- Konflikte benennen ---
    if comps["RSI"].available and comps["MACD"].available and ((rs > 15 and ms < -15) or (rs < -15 and ms > 15)):
        res.conflicts.append(f"RSI ({label(rs)}) und MACD ({label(ms)}) widersprechen sich")
    if len(bulls) >= 2 and len(bears) >= 2:
        res.conflicts.append("Bullishe: " + ", ".join(c.name for c in bulls) + " – bearishe: " + ", ".join(c.name for c in bears))
    # --- Bestätigte Wende? ---
    macd_df = ctx.macd.dropna()
    macd_up = len(macd_df) > 5 and recent(crossed_up(macd_df["macd"], macd_df["signal"]), 3)
    macd_dn = len(macd_df) > 5 and recent(crossed_down(macd_df["macd"], macd_df["signal"]), 3)
    div_bull = any(d.kind == "bullish" for d in ctx.divs); div_bear = any(d.kind == "bearish" for d in ctx.divs)
    ev = {e[0] for e in ctx.trend.events}
    bull_reversal = macd_up or div_bull or "Breakout" in ev
    bear_reversal = macd_dn or div_bear or "Breakdown" in ev
    res.decision = "ABWARTEN"; note = None
    if res.total >= st.buy_threshold and len(bulls) >= st.min_agreeing_components and rs > -15 and ms > -15:
        if ctx.trend.direction == "abwärts" and ctx.trend.strength >= 0.5 and not bull_reversal:
            note = "Erholungssignale laufen gegen einen intakten Abwärtstrend, ein bestätigtes Wendesignal (MACD-Crossover, Divergenz oder Ausbruch) fehlt noch"
        else:
            res.decision = "KAUFEN"
    elif res.total <= st.sell_threshold and len(bears) >= st.min_agreeing_components and rs < 15 and ms < 15:
        if ctx.trend.direction == "aufwärts" and ctx.trend.strength >= 0.5 and not bear_reversal:
            note = "Schwächesignale laufen gegen einen intakten Aufwärtstrend, ein bestätigtes Wendesignal fehlt noch"
        else:
            res.decision = "VERKAUFEN"
    if note: res.conflicts.append(note)
    if res.decision == "ABWARTEN" and not res.conflicts and abs(res.total) >= 20:
        res.conflicts.append("Die Richtung ist erkennbar, aber noch nicht von genügend unabhängigen Signalen bestätigt")
    # --- Technische Signalstärke (KEINE statistische Wahrscheinlichkeit, solange nicht per Backtest kalibriert) ---
    sig = [c for c in avail if abs(c.score) > 10]
    d = np.sign(res.total) or 1
    agree = sum(res.weights[c.name] for c in sig if np.sign(c.score) == d) / (sum(res.weights[c.name] for c in sig) or 1)
    strength = 100 * (0.6 * min(1, abs(res.total) / 70) + 0.4 * agree)
    res.strength = float(strength * (0.7 if res.conflicts else 1.0))
    res.summary_reasons = _top_reasons(res)


def _top_reasons(res: Analysis, k=6):
    sign = 1 if res.total >= 0 else -1
    out = []
    for c in sorted([c for c in res.components if c.available], key=lambda c: -abs(c.score * res.weights[c.name])):
        if np.sign(c.score) == sign and abs(c.score) > 10 and c.reasons:
            out += [(c.name, r) for r in c.reasons[:2]]
    return out[:k]


def _risks(res: Analysis) -> list:
    ctx, risks, comps = res.ctx, [], {c.name: c for c in res.components}
    price = float(ctx.df["Close"].iloc[-1])
    sup, rzone, inside, _, _ = nearest_zones(ctx.zones, price)
    if rzone and (rzone.low - price) / ctx.atr < 1.5:
        risks.append(f"Widerstand direkt über dem Kurs ({rzone.low:.2f}–{rzone.high:.2f}): begrenztes Aufwärtspotenzial, Rückschlagrisiko")
    if sup and (price - sup.high) / ctx.atr > 3:
        risks.append("Nächste Unterstützung ist weit entfernt – bei einem Rücksetzer gibt es wenig Halt")
    v = comps["Volumen"]
    if v.available and v.details.get("rel_volume", 1) < 0.7:
        risks.append("Schwaches Volumen – Signale sind weniger verlässlich")
    if any(d.kind == "bearish" for d in ctx.divs): risks.append("Bearishe Divergenz (RSI/MACD) erkannt")
    r = comps["RSI"]
    if r.available and r.details["value"] > 70: risks.append("RSI überkauft")
    if r.available and r.details["value"] < 30 and ctx.trend.direction == "abwärts": risks.append("RSI überverkauft im Abwärtstrend – Verkaufsdruck kann anhalten")
    if comps["MACD"].available and comps["MACD"].score < -15: risks.append("MACD bearish")
    if ctx.trend.direction == "abwärts" and ctx.trend.strength >= 0.5: risks.append("Intakter Abwärtstrend")
    atr_pct = ctx.atr / price * 100
    if atr_pct > 4: risks.append(f"Hohe Volatilität (ATR ≈ {atr_pct:.1f} % des Kurses) – Stops weiter, Positionsgröße beachten")
    if not ctx.zones: risks.append("Keine belastbaren Kurszonen erkannt")
    return risks


def _levels(res: Analysis) -> dict:
    """Einstieg/Stop/Ziele ausschließlich aus erkannten Zonen. Fehlt eine Basis -> None (Daten nicht verfügbar)."""
    ctx = res.ctx; price = float(ctx.df["Close"].iloc[-1]); a = ctx.atr
    short = res.decision == "VERKAUFEN"
    sup, rzone, inside, below, above = nearest_zones(ctx.zones, price)
    L = dict(scenario="Short/Absicherung" if short else "Long", entry=price, stop=None, target1=None, target2=None, rr=None, note="")
    if not short:
        base = inside if (inside and price >= inside.mid) else sup
        if base: L["stop"] = base.low - 0.25 * a
        tg = [z.low for z in above[:2]]
    else:
        base = inside if (inside and price < inside.mid) else rzone
        if base: L["stop"] = base.high + 0.25 * a
        tg = [z.high for z in below[:2]]
    if len(tg) > 0: L["target1"] = tg[0]
    if len(tg) > 1: L["target2"] = tg[1]
    if L["stop"] is not None and L["target1"] is not None:
        risk, reward = abs(price - L["stop"]), abs(L["target1"] - price)
        if risk > 0: L["rr"] = reward / risk
    missing = [n for n, k in (("Stop-Loss", "stop"), ("Kursziel 1", "target1")) if L[k] is None]
    L["note"] = ("Daten nicht verfügbar für: " + ", ".join(missing) + " (keine passende Zone erkannt)") if missing else \
                "Aus erkannten Zonen abgeleitet (Stop = Zonenrand ± 0,25 ATR, Ziele = nächste Gegenzonen). Keine Anlageberatung."
    return L


def _explain(res: Analysis) -> str:
    c = {x.name: x for x in res.components}
    head = {"KAUFEN": "Die technischen Signale sprechen aktuell eher für einen möglichen Einstieg.",
            "VERKAUFEN": "Die technischen Signale sprechen aktuell eher gegen einen Einstieg bzw. für anhaltende Schwäche.",
            "ABWARTEN": "Die Signale sind aktuell gemischt oder noch nicht ausreichend bestätigt."}[res.decision]
    parts = [head]
    for name in ("RSI", "MACD"):
        if c[name].available:
            parts.append(f"{name}: " + "; ".join(c[name].reasons[:3]) + ".")
    for name in ("Candlesticks", "Trend", "Unterstützung/Widerstand", "Volumen"):
        if c[name].available and abs(c[name].score) > 10:
            parts.append(f"{name}: " + "; ".join(c[name].reasons[:2]) + ".")
    if res.conflicts:
        parts.append("Einschränkungen: " + " | ".join(res.conflicts) + ".")
    if res.decision == "ABWARTEN":
        sup, rz, *_ = nearest_zones(res.ctx.zones, float(res.df["Close"].iloc[-1]))
        hints = []
        if res.total >= 0: hints += ["ein bullisher MACD-Crossover"] + ([f"ein Schlusskurs über {rz.high:.2f}"] if rz else [])
        else: hints += ["ein bearisher MACD-Crossover"] + ([f"ein Schlusskurs unter {sup.low:.2f}"] if sup else [])
        parts.append("Mehr Klarheit würde bringen: " + " oder ".join(hints) + ".")
    if res.risks:
        parts.append("Risiken: " + "; ".join(res.risks[:4]) + ".")
    return "\n\n".join(parts)
