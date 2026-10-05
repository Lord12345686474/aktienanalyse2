"""CLI: python -m stock_analyzer.main AAPL  (oder Firmenname in Anführungszeichen)"""
import argparse, sys
from .config import Settings
from .data.market_data import YFinanceProvider, DataUnavailable, NO_DATA_MSG, is_stale
from .analysis.signal_engine import analyze
from .analysis.multi_timeframe import analyze_multi

ICON = {"KAUFEN": "🟢", "ABWARTEN": "🟡", "VERKAUFEN": "🔴"}


def render(sym, a, mtf=None) -> str:
    L = [f"Analyse: {sym}", f"Gesamtsignal: {a.label.upper()}   Score: {a.total:+.0f}/100",
         f"Entscheidung: {ICON[a.decision]} {a.decision}   Technische Signalstärke: {a.strength:.0f}/100 (heuristisch, keine Wahrscheinlichkeit)", ""]
    for c in a.components:
        L.append(f"{c.name}: {c.score:+.0f} ({'Daten nicht verfügbar' if not c.available else ''}) " if not c.available else f"{c.name}: {c.score:+.0f}")
        L += [f"   - {r}" for r in c.reasons[:4]]
    if mtf and mtf.per_tf:
        L += ["", "Zeiteinheiten: " + ", ".join(f"{tf} {x.total:+.0f}" for tf, x in mtf.per_tf.items()), mtf.headline]
    lv = a.levels
    f = lambda v: "Daten nicht verfügbar" if v is None else f"{v:.2f}"
    L += ["", f"Szenario {lv['scenario']}: Einstieg {f(lv['entry'])} | Stop {f(lv['stop'])} | Ziel 1 {f(lv['target1'])} | Ziel 2 {f(lv['target2'])} | CRV {f(lv['rr'])}", lv["note"], "", a.explanation]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("symbol"); ap.add_argument("--interval", default="1d")
    ap.add_argument("--multi", action="store_true"); args = ap.parse_args(argv)
    prov, st = YFinanceProvider(), Settings()
    try:
        sym = prov.resolve(args.symbol); df = prov.history(sym, args.interval)
    except DataUnavailable as e:
        print(NO_DATA_MSG, f"({e})"); return 1
    if is_stale(df, args.interval): print("⚠ Daten sind nicht aktuell (letzte Kerze:", df.index[-1], ")")
    a = analyze(df, st)
    if not a.ok: print(a.message); return 1
    print(render(sym, a, analyze_multi(prov, sym, st) if args.multi else None)); return 0


if __name__ == "__main__":
    sys.exit(main())
