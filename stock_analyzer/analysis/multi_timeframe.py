"""Mehrere Zeiteinheiten: jeder Timeframe wird einzeln analysiert, dann gewichtet kombiniert.

Widersprüche zwischen Timeframes führen NIE zu einem stumpfen 'Kaufen', sondern zu ABWARTEN mit Erklärung
(z. B. 'kurzfristige Schwäche innerhalb eines übergeordneten Aufwärtstrends').
"""
from dataclasses import dataclass, field
from ..config import Settings
from ..data.market_data import DataUnavailable, MarketDataProvider, is_stale
from .signal_engine import analyze, Analysis

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1wk"]
HORIZON_WEIGHTS = {
    "long":  {"5m": 0.03, "15m": 0.04, "1h": 0.08, "4h": 0.15, "1d": 0.35, "1wk": 0.35},
    "short": {"5m": 0.15, "15m": 0.20, "1h": 0.25, "4h": 0.20, "1d": 0.15, "1wk": 0.05},
}
HIGHER = {"1d", "1wk"}


@dataclass
class MTFResult:
    per_tf: dict = field(default_factory=dict)       # tf -> Analysis
    errors: dict = field(default_factory=dict)       # tf -> Meldung
    stale: dict = field(default_factory=dict)
    total: float = 0.0
    decision: str = "ABWARTEN"
    headline: str = ""
    conflicts: list = field(default_factory=list)
    primary: str = "1d"


def _bias(a: Analysis) -> int:
    return 1 if a.total >= 12 else -1 if a.total <= -12 else 0


def analyze_multi(provider: MarketDataProvider, symbol: str, st: Settings, timeframes=("1h", "4h", "1d", "1wk")) -> MTFResult:
    out = MTFResult()
    for tf in timeframes:
        try:
            df = provider.history(symbol, tf)
            out.stale[tf] = is_stale(df, tf)
            a = analyze(df, st)
            (out.per_tf.__setitem__(tf, a) if a.ok else out.errors.__setitem__(tf, a.message))
        except DataUnavailable as e:
            out.errors[tf] = str(e)
    return combine(out, st)


def combine(out: MTFResult, st: Settings) -> MTFResult:
    if not out.per_tf:
        out.headline = "Keine aktuellen Marktdaten verfügbar."; return out
    w = {tf: HORIZON_WEIGHTS[st.horizon].get(tf, 0.05) for tf in out.per_tf}
    s = sum(w.values())
    out.total = sum(w[tf] * a.total for tf, a in out.per_tf.items()) / s
    out.primary = "1d" if "1d" in out.per_tf else sorted(out.per_tf, key=lambda t: -w[t])[0]
    prim = out.per_tf[out.primary]
    bias = {tf: _bias(a) for tf, a in out.per_tf.items()}
    relevant = {tf: b for tf, b in bias.items() if w[tf] >= 0.08}
    out.decision = prim.decision
    if 1 in relevant.values() and -1 in relevant.values():
        hi = [b for tf, b in bias.items() if tf in HIGHER]; lo = [b for tf, b in bias.items() if tf not in HIGHER]
        hi_s, lo_s = sum(hi), sum(lo)
        if hi and lo and hi_s > 0 > lo_s:
            out.headline = "Gemischtes Signal – kurzfristige Schwäche innerhalb eines übergeordneten Aufwärtstrends."
        elif hi and lo and hi_s < 0 < lo_s:
            out.headline = "Gemischtes Signal – kurzfristige Erholung innerhalb eines übergeordneten Abwärtstrends."
        else:
            out.headline = "Gemischtes Signal – die Zeiteinheiten widersprechen sich."
        out.conflicts.append(out.headline); out.decision = "ABWARTEN"
    else:
        out.headline = "Die Zeiteinheiten sind weitgehend im Einklang." if out.decision != "ABWARTEN" else prim.label
    # Der Hauptzeitrahmen darf nur dann KAUFEN/VERKAUFEN liefern, wenn auch der kombinierte Score zustimmt
    if out.decision == "KAUFEN" and out.total < st.buy_threshold * 0.6: out.decision = "ABWARTEN"
    if out.decision == "VERKAUFEN" and out.total > st.sell_threshold * 0.6: out.decision = "ABWARTEN"
    return out
