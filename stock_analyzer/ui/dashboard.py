"""Streamlit-Dashboard. Start:  streamlit run stock_analyzer/ui/dashboard.py   (aus dem Ordner ÜBER stock_analyzer/)

Designentscheidung Streamlit + Plotly: reine Python-Lösung, interaktive Candlestick-Charts mit Subplots
(Preis / Volumen / RSI / MACD teilen die X-Achse), schnell erweiterbar. Volumen liegt als eigene Zeile direkt
unter dem Preischart (bessere Lesbarkeit als Overlay).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer.config import Settings
from stock_analyzer.data.market_data import YFinanceProvider, DataUnavailable, NO_DATA_MSG, is_stale
from stock_analyzer.analysis.signal_engine import analyze
from stock_analyzer.analysis.scoring import label
from stock_analyzer.analysis.multi_timeframe import analyze_multi, TIMEFRAMES
from stock_analyzer.backtesting.backtest import run_all

ICON = {"KAUFEN": "🟢", "ABWARTEN": "🟡", "VERKAUFEN": "🔴"}
na = lambda v, f="{:,.2f}": "Daten nicht verfügbar" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)
st.set_page_config(page_title="Technischer Aktienanalyse-Assistent", layout="wide")
provider = YFinanceProvider()


@st.cache_data(ttl=300, show_spinner=False)
def load(symbol, tf):
    return provider.history(symbol, tf)


def build_settings() -> Settings:
    s = Settings()
    with st.sidebar:
        st.header("Einstellungen")
        s.rsi_period = st.number_input("RSI-Periode", 2, 50, 14)
        c1, c2, c3 = st.columns(3)
        s.macd_fast = c1.number_input("MACD fast", 2, 50, 12); s.macd_slow = c2.number_input("slow", 3, 100, 26); s.macd_signal = c3.number_input("signal", 2, 50, 9)
        sm = st.text_input("SMA-Perioden", "20,50,200"); em = st.text_input("EMA-Perioden", "20,50")
        try:
            s.sma_periods = tuple(int(x) for x in sm.split(",")); s.ema_periods = tuple(int(x) for x in em.split(","))
        except ValueError:
            st.warning("Ungültige Perioden – Standard wird verwendet.")
        s.horizon = "long" if st.radio("Anlagehorizont", ["Langfristig", "Kurzfristig"]) == "Langfristig" else "short"
        st.subheader("Gewichtung (%)")
        s.weights = {k: st.slider(k, 0, 50, int(v)) for k, v in s.weights.items()}
        s.buy_threshold = st.slider("Schwelle KAUFEN", 10, 60, int(s.buy_threshold)); s.sell_threshold = -st.slider("Schwelle VERKAUFEN", 10, 60, int(-s.sell_threshold))
    return s


def chart(a, tf: str, show_bars: int):
    ctx, df = a.ctx, a.df.iloc[-show_bars:]
    off = len(a.df) - len(df); idx = df.index
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.12, 0.19, 0.19])
    fig.add_trace(go.Candlestick(x=idx, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Kurs"), 1, 1)
    for col in ctx.ma.columns:
        fig.add_trace(go.Scatter(x=idx, y=ctx.ma[col].iloc[off:], name=col, line=dict(width=1)), 1, 1)
    lo, hi = df.Low.min(), df.High.max()
    for z in ctx.zones:
        if z.high >= lo and z.low <= hi:
            col = "rgba(0,160,80,0.15)" if z.high < df.Close.iloc[-1] else "rgba(220,50,50,0.15)"
            fig.add_hrect(y0=z.low, y1=z.high, fillcolor=col, line_width=0, row=1, col=1)
    for p in ctx.patterns:
        if p.pos >= off and p.bias != "neutral":
            up = p.bias == "bullish"; row = a.df.iloc[p.pos]
            fig.add_trace(go.Scatter(x=[p.time], y=[row.Low * 0.995 if up else row.High * 1.005], mode="markers+text", text=[p.name], textposition="bottom center" if up else "top center",
                                     marker=dict(symbol="triangle-up" if up else "triangle-down", size=9, color="green" if up else "red"), showlegend=False), 1, 1)
    vcol = np.where(df.Close >= df.Open, "rgba(0,160,80,0.5)", "rgba(220,50,50,0.5)")
    fig.add_trace(go.Bar(x=idx, y=df.Volume, marker_color=vcol, name="Volumen"), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=ctx.rsi.iloc[off:], name="RSI", line=dict(color="purple")), 3, 1)
    for lvl, dash in ((30, "dot"), (50, "dash"), (70, "dot")):
        fig.add_hline(y=lvl, line_dash=dash, line_color="gray", row=3, col=1)
    m = ctx.macd.iloc[off:]
    fig.add_trace(go.Scatter(x=idx, y=m["macd"], name="MACD", line=dict(color="blue")), 4, 1)
    fig.add_trace(go.Scatter(x=idx, y=m["signal"], name="Signal", line=dict(color="orange")), 4, 1)
    fig.add_trace(go.Bar(x=idx, y=m["hist"], name="Histogramm", marker_color=np.where(m["hist"] >= 0, "rgba(0,160,80,0.5)", "rgba(220,50,50,0.5)")), 4, 1)
    fig.add_hline(y=0, line_color="gray", row=4, col=1)
    up = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1)); dn = (m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))
    fig.add_trace(go.Scatter(x=idx[up.to_numpy()], y=m["macd"][up], mode="markers", marker=dict(symbol="triangle-up", color="green", size=9), name="Bull-Cross"), 4, 1)
    fig.add_trace(go.Scatter(x=idx[dn.to_numpy()], y=m["macd"][dn], mode="markers", marker=dict(symbol="triangle-down", color="red", size=9), name="Bear-Cross"), 4, 1)
    for d in ctx.divs:                                               # Divergenzen als Linien in Preis- und Indikatorchart
        col = "green" if d.kind == "bullish" else "red"
        if d.pos1 >= off:
            if d.indicator == "RSI":
                fig.add_trace(go.Scatter(x=[d.time1, d.time2], y=[d.ind1, d.ind2], mode="lines+markers", line=dict(color=col, dash="dash"), name=f"{d.kind} RSI-Div."), 3, 1)
            else:
                fig.add_trace(go.Scatter(x=[d.time1, d.time2], y=[d.ind1, d.ind2], mode="lines+markers", line=dict(color=col, dash="dash"), name=f"{d.kind} MACD-Div."), 4, 1)
            fig.add_trace(go.Scatter(x=[d.time1, d.time2], y=[d.price1, d.price2], mode="lines", line=dict(color=col, dash="dash"), showlegend=False), 1, 1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    if tf in ("1d", "1wk"):
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] if tf == "1d" else [])
    return fig


def rating_panel(a):
    st.subheader("TECHNISCHE BEWERTUNG")
    st.markdown(f"### {ICON[a.decision]} {a.label}")
    st.metric("Score", f"{a.total:+.0f}/100"); st.progress(int((a.total + 100) / 2))
    for c in a.components:
        st.write(f"**{c.name}:** " + ("Daten nicht verfügbar" if not c.available else f"{label(c.score)} ({c.score:+.0f})"))
    st.markdown(f"#### Gesamtentscheidung: {ICON[a.decision]} {a.decision}")
    st.caption(f"Technische Signalstärke: {a.strength:.0f}/100 – heuristischer Wert, keine statistische Wahrscheinlichkeit (nicht per Backtest kalibriert).")
    if a.conflicts: st.warning("Einschränkungen:\n\n- " + "\n- ".join(a.conflicts))
    st.write(a.explanation)
    if a.risks: st.error("Risiken:\n\n- " + "\n- ".join(a.risks))
    lv = a.levels
    st.markdown(f"**Szenario {lv['scenario']}** – Einstieg {na(lv['entry'])} · Stop {na(lv['stop'])} · Ziel 1 {na(lv['target1'])} · Ziel 2 {na(lv['target2'])} · CRV {na(lv['rr'], '{:.1f}')}")
    st.caption(lv["note"])


def main():
    s = build_settings()
    st.title("📈 Technischer Aktienanalyse-Assistent")
    q = st.text_input("Aktie (Ticker oder Name)", "AAPL")
    tf = st.selectbox("Zeiteinheit (Hauptchart)", TIMEFRAMES, index=4)
    if not q: return
    try:
        sym = provider.resolve(q); df = load(sym, tf)
    except DataUnavailable as e:
        st.error(f"{NO_DATA_MSG} ({e})"); return
    if is_stale(df, tf): st.warning(f"Die Daten sind nicht aktuell (letzte Kerze: {df.index[-1]}).")
    quote = provider.quote(sym)
    k = st.columns(6)
    k[0].metric(quote.name or sym, na(quote.last_price), None if quote.day_change_pct is None else f"{quote.day_change_pct:+.2f} %")
    k[1].metric("Marktkap.", na(quote.market_cap, "{:,.0f}")); k[2].metric("Volumen", na(quote.volume, "{:,.0f}"))
    k[3].metric("52W-Hoch", na(quote.high_52w)); k[4].metric("52W-Tief", na(quote.low_52w)); k[5].metric("Letzte Kerze", str(df.index[-1])[:16])
    a = analyze(df, s)
    if not a.ok: st.error(a.message); return
    left, right = st.columns([3, 1.4])
    with left:
        bars = st.slider("Sichtbare Kerzen", 60, min(len(df), 600), min(len(df), 180))
        st.plotly_chart(chart(a, tf, bars))
    with right: rating_panel(a)
    t1, t2, t3, t4, t5 = st.tabs(["RSI-Details", "MACD-Details", "Candlesticks", "Zeiteinheiten", "Backtest"])
    with t1:
        c = a.component("RSI"); st.write("\n".join(f"- {r}" for r in c.reasons)); st.json(c.details) if c.available else None
    with t2:
        c = a.component("MACD"); st.write("\n".join(f"- {r}" for r in c.reasons)); st.json(c.details) if c.available else None
    with t3:
        pats = a.component("Candlesticks").details.get("patterns", [])
        st.dataframe(pd.DataFrame(pats)[["time", "name", "bias", "strength", "at_level", "confirmed", "text"]] if pats else pd.DataFrame({"Info": ["Keine Formation in den letzten 3 Kerzen"]}))
    with t4:
        if st.button("Alle Zeiteinheiten analysieren"):
            m = analyze_multi(provider, sym, s)
            st.subheader(m.headline); st.write(f"Kombinierter Score: {m.total:+.0f} → {ICON[m.decision]} {m.decision}")
            st.dataframe(pd.DataFrame([dict(Zeiteinheit=t, Score=round(x.total), Signal=x.label, Entscheidung=x.decision) for t, x in m.per_tf.items()]))
            for t, msg in m.errors.items(): st.caption(f"{t}: {msg}")
    with t5:
        full = st.checkbox("Gesamtstrategie (langsam)", False); fee = st.number_input("Gebühr pro Seite (%)", 0.0, 1.0, 0.1) / 100
        if st.button("Backtest starten"):
            res = run_all(df, s, tf, include_full=full, fee=fee, full_step=3)
            rows = [dict(Strategie=r.name, Trades=r.metrics["trades"], Trefferquote=na(r.metrics["hit_rate"], "{:.0%}"), Ø_Gewinn=na(r.metrics["avg_win"], "{:.1%}"), Ø_Verlust=na(r.metrics["avg_loss"], "{:.1%}"),
                         ProfitFactor=na(r.metrics["profit_factor"], "{:.2f}"), MaxDrawdown=f"{r.metrics['max_drawdown']:.1%}", Sharpe=na(r.metrics["sharpe"], "{:.2f}"), Rendite=f"{r.metrics['total_return']:.1%}", BuyHold=f"{r.metrics['buy_hold_return']:.1%}") for r in res]
            st.dataframe(pd.DataFrame(rows))
            fig = go.Figure([go.Scatter(x=r.equity.index, y=r.equity, name=r.name) for r in res] + [go.Scatter(x=df.index, y=res[0].buy_hold, name="Buy & Hold", line=dict(dash="dot", color="black"))])
            st.plotly_chart(fig); st.caption("Ausführung zum Open der Folgekerze, inkl. Gebühren, ohne Look-Ahead. Vergangene Ergebnisse sind keine Garantie; bei wenigen Trades statistisch schwach.")
    st.caption("Keine Anlageberatung. Technische Signale sind Wahrscheinlichkeitsindizien, keine Prognosen.")


main()
