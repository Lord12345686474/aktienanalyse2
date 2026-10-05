import unittest, numpy as np, pandas as pd
from stock_analyzer.indicators.rsi import rsi
from stock_analyzer.indicators.macd import macd, ema
from stock_analyzer.patterns.candlestick_patterns import detect_patterns
from stock_analyzer.analysis.swings import find_pivots
from stock_analyzer.analysis.signal_engine import analyze
from stock_analyzer.backtesting.backtest import run_backtest, strategy_signals
from stock_analyzer.config import Settings
from stock_analyzer.data.market_data import clean_ohlcv, DataUnavailable
from .synthetic import make_df

# Klassisches Lehrbuch-Beispiel (StockCharts "RSI" Kalkulationstabelle), RSI(14)
SC_CLOSE = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
            46.28, 46.00, 46.03, 46.41, 46.22, 45.64]


def ref_rsi(closes, n=14):
    """Unabhängige Referenz in reinem Python (Wilder)."""
    g = [max(b - a, 0) for a, b in zip(closes, closes[1:])]; l = [max(a - b, 0) for a, b in zip(closes, closes[1:])]
    ag, al = sum(g[:n]) / n, sum(l[:n]) / n; out = [100 - 100 / (1 + ag / al)]
    for i in range(n, len(g)):
        ag, al = (ag * (n - 1) + g[i]) / n, (al * (n - 1) + l[i]) / n; out.append(100 - 100 / (1 + ag / al))
    return out


class TestIndicators(unittest.TestCase):
    def test_rsi_matches_reference(self):
        df = make_df(200); ours = rsi(df["Close"], 14).dropna().to_numpy()
        np.testing.assert_allclose(ours, ref_rsi(df["Close"].tolist()), atol=1e-9)

    def test_rsi_textbook(self):
        v = rsi(pd.Series(SC_CLOSE), 14).dropna().iloc[0]
        self.assertAlmostEqual(v, 70.5, delta=0.5)       # StockCharts nennt 70,53 (gerundete Zwischenwerte)

    def test_rsi_bounds_and_extremes(self):
        up = rsi(pd.Series(np.arange(1, 60, dtype=float)), 14).dropna()
        self.assertTrue((up == 100).all())
        r = rsi(make_df(300)["Close"]).dropna(); self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_macd_recursion(self):
        s = make_df(120)["Close"]; m = macd(s)
        a12, a26, sig, prev12, prev26, ps = s.iloc[0], s.iloc[0], None, None, None, None
        e12, e26, line = [s.iloc[0]], [s.iloc[0]], []
        for x in s.iloc[1:]:
            e12.append(e12[-1] + 2 / 13 * (x - e12[-1])); e26.append(e26[-1] + 2 / 27 * (x - e26[-1]))
        line = np.array(e12) - np.array(e26); sg = [line[0]]
        for x in line[1:]: sg.append(sg[-1] + 2 / 10 * (x - sg[-1]))
        np.testing.assert_allclose(m["macd"].dropna().to_numpy(), line[25:], atol=1e-9)
        np.testing.assert_allclose(m["signal"].dropna().to_numpy(), np.array(sg)[25:], atol=1e-9)
        np.testing.assert_allclose(m["hist"].dropna().to_numpy(), (line - np.array(sg))[25:], atol=1e-9)

    def test_macd_constant_series_is_zero(self):
        m = macd(pd.Series([50.0] * 100)).dropna(); self.assertTrue((m.abs() < 1e-12).all().all())

    def test_causality_no_lookahead(self):
        df = make_df(300); full_r, full_m = rsi(df["Close"]), macd(df["Close"])
        cut = 200; r2, m2 = rsi(df["Close"].iloc[:cut]), macd(df["Close"].iloc[:cut])
        np.testing.assert_allclose(full_r.iloc[:cut].dropna(), r2.dropna(), atol=1e-12)
        np.testing.assert_allclose(full_m.iloc[:cut].dropna(), m2.dropna(), atol=1e-12)

    def test_pivots_only_confirmed(self):
        df = make_df(100); h, l = find_pivots(df, 5)
        self.assertTrue(all(p <= len(df) - 1 - 5 for p, _ in h + l))


class TestPatterns(unittest.TestCase):
    def _df(self, rows):
        base = [(110 - i, 111 - i, 108 - i, 109 - i) for i in range(10)]       # klarer Abwärtstrend
        r = base + rows; idx = pd.bdate_range("2024-01-01", periods=len(r))
        return pd.DataFrame(r, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1e6)

    def test_bullish_engulfing(self):
        df = self._df([(100.0, 100.5, 97.5, 98.0), (97.5, 102.5, 97.0, 102.0)])
        self.assertIn("Bullish Engulfing", [p.name for p in detect_patterns(df) if p.pos == len(df) - 1])

    def test_hammer_in_downtrend(self):
        df = self._df([(100.0, 100.3, 95.0, 100.2)])
        self.assertIn("Hammer", [p.name for p in detect_patterns(df)])

    def test_doji(self):
        df = self._df([(100.0, 101.0, 99.0, 100.02)])
        self.assertIn("Doji", [p.name for p in detect_patterns(df)])


class TestEngine(unittest.TestCase):
    def test_analyze_runs_and_is_consistent(self):
        a = analyze(make_df(500, seed=3), Settings())
        self.assertTrue(a.ok); self.assertIn(a.decision, ("KAUFEN", "ABWARTEN", "VERKAUFEN"))
        self.assertTrue(-100 <= a.total <= 100); self.assertAlmostEqual(sum(a.weights.values()), 1.0)
        self.assertTrue(a.explanation)

    def test_too_little_data(self):
        self.assertFalse(analyze(make_df(30)).ok)

    def test_never_buy_on_single_indicator(self):
        # Gewicht komplett auf RSI -> darf trotzdem nie KAUFEN ohne >= 3 übereinstimmende Komponenten
        st = Settings(weights={"RSI": 100, "MACD": 0, "Candlesticks": 0, "Trend": 0, "Unterstützung/Widerstand": 0, "Volumen": 0, "Gleitende Durchschnitte": 0})
        for seed in range(15):
            a = analyze(make_df(400, seed=seed), st)
            if a.decision == "KAUFEN":
                self.assertGreaterEqual(len([c for c in a.components if c.score > 15]), 3)

    def test_clean_rejects_empty(self):
        with self.assertRaises(DataUnavailable): clean_ohlcv(pd.DataFrame())

    def test_signal_unchanged_by_future_data(self):
        df = make_df(500, seed=5); t = 400
        a1 = analyze(df.iloc[:t]); df2 = df.copy(); df2.iloc[t:, :4] *= 3          # Zukunft drastisch verändert
        a2 = analyze(df2.iloc[:t]); self.assertEqual(a1.total, a2.total)


class TestBacktest(unittest.TestCase):
    def test_executes_next_open_and_metrics(self):
        df = make_df(400, seed=2); sig = strategy_signals(df, "MACD", Settings())
        res = run_backtest(df, sig, "MACD")
        self.assertGreater(res.metrics["trades"], 0)
        if len(res.trades):
            t0 = res.trades.iloc[0]; i = df.index.get_loc(t0.entry_time)
            self.assertFalse(sig.iloc[i - 2])  # Signal entsteht erst auf Bar i-1 ...
            self.assertTrue(sig.iloc[i - 1]);  # ... Ausführung zum Open von Bar i
            self.assertAlmostEqual(t0.entry, df["Open"].iloc[i] * 1.001)
        self.assertLessEqual(res.metrics["max_drawdown"], 0)

    def test_always_in_equals_buy_and_hold_without_fees(self):
        df = make_df(300, seed=4); res = run_backtest(df, pd.Series(True, index=df.index), "BH", fee=0.0)
        # Einstieg erst zum Open von Bar 1 -> kleine Abweichung zu B&H ab Close 0 erwartet
        self.assertAlmostEqual(res.equity.iloc[-1], df["Close"].iloc[-1] / df["Open"].iloc[1], places=9)


if __name__ == "__main__":
    unittest.main()
