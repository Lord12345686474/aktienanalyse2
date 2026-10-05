"""Backtesting ohne Look-Ahead-Bias.

Regeln, die Leakage verhindern:
  * Signal wird am SCHLUSS von Bar t berechnet, die Ausführung erfolgt zum ERÖFFNUNGSKURS von Bar t+1.
  * Indikatoren (EMA, Wilder-RSI, Rolling-Mittel) sind kausal. Pivots werden erst n Bars später bestätigt.
  * Die Gesamtstrategie ruft `analyze()` pro Bar nur mit df.iloc[:t+1] auf – die Zukunft ist nicht sichtbar.
  * Gebühren/Slippage pro Seite einstellbar (Standard 0,1 %). Nur Long-Strategien (Aktienkauf).
Kennzahlen sind Vergangenheitswerte und keine Garantie für die Zukunft; kleine Trade-Zahlen sind statistisch schwach.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ..config import Settings
from ..indicators.rsi import rsi as calc_rsi
from ..indicators.macd import macd as calc_macd
from ..indicators.moving_averages import crossed_up, crossed_down
from ..patterns.candlestick_patterns import detect_patterns
from ..analysis.signal_engine import analyze

PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1h": 252 * 7, "4h": 252 * 2, "15m": 252 * 26, "5m": 252 * 78}


@dataclass
class BTResult:
    name: str
    trades: pd.DataFrame
    equity: pd.Series
    buy_hold: pd.Series
    metrics: dict = field(default_factory=dict)


def _state_machine(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    pos, out = False, []
    for e, x in zip(entry.fillna(False).to_numpy(), exit_.fillna(False).to_numpy()):
        pos = (not x) if pos else bool(e)
        out.append(pos)
    return pd.Series(out, index=entry.index)


def strategy_signals(df: pd.DataFrame, name: str, st: Settings) -> pd.Series:
    """Gewünschte Position (True = Long) am Schlusskurs jeder Bar – nur aus Daten <= t."""
    r, m = calc_rsi(df["Close"], st.rsi_period), calc_macd(df["Close"], st.macd_fast, st.macd_slow, st.macd_signal)
    m_up, m_dn = crossed_up(m["macd"], m["signal"]), crossed_down(m["macd"], m["signal"])
    if name == "RSI":
        return _state_machine(crossed_up(r, pd.Series(30.0, index=r.index)), (r > 70) | (r < 45) & (r.shift(1) >= 45))
    if name == "MACD":
        return _state_machine(m_up, m_dn)
    rsi_recovering = (r.rolling(10).min() < 40) & (r > r.shift(2))
    hist_better = m["hist"] > m["hist"].shift(1)
    combo = rsi_recovering & hist_better & m_up.rolling(3).max().astype(bool)
    if name == "RSI+MACD":
        return _state_machine(combo, m_dn | (r > 72))
    if name == "Candlestick+RSI+MACD":
        bull = pd.Series(False, index=df.index)
        for p in detect_patterns(df):
            if p.bias == "bullish": bull.iloc[p.pos] = True
        bull3 = bull.rolling(3).max().astype(bool)
        loose = rsi_recovering & (m["hist"] > m["hist"].shift(1)) & (m["macd"] > m["signal"])
        return _state_machine(bull3 & loose, m_dn | (r > 72))
    raise ValueError(name)


def full_strategy_signals(df: pd.DataFrame, st: Settings, warmup: int = 220, window: int = 400, step: int = 1) -> pd.Series:
    """Gesamtstrategie: pro Bar komplette Analyse mit ausschließlich vergangenen Daten (Expanding/Rolling-Window)."""
    want, pos = [], False
    for t in range(len(df)):
        if t < warmup or (t - warmup) % step:
            want.append(pos); continue
        a = analyze(df.iloc[max(0, t + 1 - window): t + 1], st)
        if a.ok:
            pos = (a.decision == "KAUFEN") if not pos else (a.decision != "VERKAUFEN" and a.total > 0)
        want.append(pos)
    return pd.Series(want, index=df.index)


def run_backtest(df: pd.DataFrame, want_long: pd.Series, name: str, fee: float = 0.001, interval: str = "1d") -> BTResult:
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    w = want_long.reindex(df.index).fillna(False).to_numpy()
    equity, eq, in_pos, entry_px, entry_i, trades = [1.0], 1.0, False, 0.0, 0, []
    for i in range(1, len(df)):
        # Entscheidung stammt vom Schluss der Vorbar (i-1) -> Ausführung zum Open von Bar i
        if not in_pos and w[i - 1]:
            in_pos, entry_px, entry_i = True, o[i] * (1 + fee), i
            eq *= (c[i] / entry_px)
        elif in_pos and not w[i - 1]:
            exit_px = o[i] * (1 - fee)
            eq *= exit_px / c[i - 1]
            trades.append(dict(entry_time=df.index[entry_i], exit_time=df.index[i], entry=entry_px, exit=exit_px, ret=exit_px / entry_px - 1))
            in_pos = False
        elif in_pos:
            eq *= c[i] / c[i - 1]
        equity.append(eq)
    if in_pos:                                                   # offene Position zum letzten Schluss bewerten
        trades.append(dict(entry_time=df.index[entry_i], exit_time=df.index[-1], entry=entry_px, exit=c[-1], ret=c[-1] / entry_px - 1))
    eqs = pd.Series(equity, index=df.index)
    bh = df["Close"] / df["Close"].iloc[0]
    tr = pd.DataFrame(trades, columns=["entry_time", "exit_time", "entry", "exit", "ret"])
    return BTResult(name, tr, eqs, bh, _metrics(tr, eqs, bh, interval))


def _metrics(tr: pd.DataFrame, eq: pd.Series, bh: pd.Series, interval: str) -> dict:
    n = len(tr); wins, losses = tr[tr.ret > 0].ret, tr[tr.ret <= 0].ret
    rets = eq.pct_change().dropna()
    ppy = PERIODS_PER_YEAR.get(interval, 252)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ppy)) if len(rets) > 20 and rets.std() > 0 else None
    return dict(
        trades=n,
        hit_rate=float(len(wins) / n) if n else None,
        avg_win=float(wins.mean()) if len(wins) else None,
        avg_loss=float(losses.mean()) if len(losses) else None,
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        max_drawdown=float((eq / eq.cummax() - 1).min()),
        sharpe=sharpe,
        total_return=float(eq.iloc[-1] - 1),
        buy_hold_return=float(bh.iloc[-1] - 1),
        buy_hold_max_drawdown=float((bh / bh.cummax() - 1).min()),
        note="Wenige Trades (<30): statistisch kaum belastbar." if n < 30 else "",
    )


def run_all(df: pd.DataFrame, st: Settings, interval="1d", include_full=True, fee=0.001, full_step=1) -> list[BTResult]:
    res = [run_backtest(df, strategy_signals(df, n, st), n, fee, interval) for n in ("RSI", "MACD", "RSI+MACD", "Candlestick+RSI+MACD")]
    if include_full:
        res.append(run_backtest(df, full_strategy_signals(df, st, step=full_step), "Gesamtstrategie", fee, interval))
    return res
