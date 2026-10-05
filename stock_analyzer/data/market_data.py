"""Datenschicht. Der Anbieter ist über das Interface `MarketDataProvider` austauschbar.

Designentscheidung: Alle Anbieter liefern dasselbe Format (DataFrame mit Spalten
Open, High, Low, Close, Volume und DatetimeIndex). Der Rest des Programms kennt
keinen Anbieter. Fehlen Daten, wird `DataUnavailable` geworfen – es werden NIE
Werte erfunden oder geschätzt.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

NO_DATA_MSG = "Keine aktuellen Marktdaten verfügbar."
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Maximale Historie je Intervall bei yfinance (Anbieter-Limits)
YF_PERIOD = {"5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d", "1d": "5y", "1wk": "10y"}
# Maximal tolerierbares Alter der letzten Kerze, bevor wir "nicht aktuell" warnen
MAX_AGE = {"5m": pd.Timedelta(days=4), "15m": pd.Timedelta(days=4), "1h": pd.Timedelta(days=5),
           "4h": pd.Timedelta(days=5), "1d": pd.Timedelta(days=6), "1wk": pd.Timedelta(days=14)}


class DataUnavailable(Exception):
    """Keine (oder keine brauchbaren) Marktdaten."""


@dataclass
class QuoteInfo:
    """Kennzahlen zur Anzeige. None = 'Daten nicht verfügbar'. Fließen NICHT in den Score ein."""
    symbol: str
    name: str | None = None
    last_price: float | None = None
    day_change_pct: float | None = None
    market_cap: float | None = None
    volume: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    currency: str | None = None


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validiert/bereinigt Rohdaten, ohne Werte zu erfinden (fehlerhafte Zeilen werden gelöscht)."""
    if df is None or df.empty:
        raise DataUnavailable(NO_DATA_MSG)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataUnavailable(f"Spalten fehlen: {missing}")
    df = df[OHLCV].astype(float).dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise DataUnavailable(NO_DATA_MSG)
    return df


def is_stale(df: pd.DataFrame, interval: str) -> bool:
    """True, wenn die letzte Kerze deutlich älter ist als erwartet (Wochenenden berücksichtigt)."""
    last = df.index[-1]
    now = pd.Timestamp.now(tz=last.tz) if last.tzinfo else pd.Timestamp.now()
    return (now - last) > MAX_AGE.get(interval, pd.Timedelta(days=6))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregiert OHLCV (z. B. 1h -> 4h)."""
    out = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Close"])


class MarketDataProvider(ABC):
    @abstractmethod
    def history(self, symbol: str, interval: str = "1d") -> pd.DataFrame: ...
    @abstractmethod
    def quote(self, symbol: str) -> QuoteInfo: ...
    def resolve(self, query: str) -> str:
        """Firmenname -> Ticker. Standard: Eingabe unverändert als Ticker verwenden."""
        return query.strip().upper()


class YFinanceProvider(MarketDataProvider):
    """Standardanbieter (inoffizielle Yahoo-Daten; für Produktivbetrieb ggf. lizenzierte API nutzen)."""

    def resolve(self, query: str) -> str:
        q = query.strip()
        if not q:
            raise DataUnavailable("Leere Eingabe.")
        if " " not in q and len(q) <= 8:        # sieht wie ein Ticker aus
            return q.upper()
        try:
            import yfinance as yf
            hits = yf.Search(q, max_results=5).quotes
            equities = [h for h in hits if h.get("quoteType") == "EQUITY"] or hits
            return equities[0]["symbol"]
        except Exception as exc:                 # Netzwerk, API-Änderung ...
            raise DataUnavailable(f"Name '{q}' konnte nicht aufgelöst werden ({exc}).") from exc

    def history(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
            fetch_iv = "1h" if interval == "4h" else interval
            raw = yf.download(symbol, period=YF_PERIOD[interval], interval=fetch_iv,
                              auto_adjust=True, progress=False)
        except Exception as exc:
            raise DataUnavailable(f"{NO_DATA_MSG} ({exc})") from exc
        df = clean_ohlcv(raw)
        return resample_ohlcv(df, "4h") if interval == "4h" else df

    def quote(self, symbol: str) -> QuoteInfo:
        q = QuoteInfo(symbol=symbol)
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            fi = t.fast_info
            g = lambda k: (fi[k] if k in fi.keys() else None)   # fehlend -> None, nie schätzen
            q.last_price, prev = g("last_price"), g("previous_close")
            q.day_change_pct = (q.last_price / prev - 1) * 100 if q.last_price and prev else None
            q.market_cap, q.volume = g("market_cap"), g("last_volume")
            q.high_52w, q.low_52w, q.currency = g("year_high"), g("year_low"), g("currency")
            try:
                info = t.info; q.name = info.get("longName") or info.get("shortName")
            except Exception:
                pass
        except Exception:
            pass
        return q


class CSVProvider(MarketDataProvider):
    """Offline-/Test-Anbieter: liest <ordner>/<SYMBOL>_<interval>.csv (Date,Open,High,Low,Close,Volume)."""
    def __init__(self, folder: str): self.folder = folder
    def history(self, symbol, interval="1d"):
        import os
        path = os.path.join(self.folder, f"{symbol}_{interval}.csv")
        if not os.path.exists(path):
            raise DataUnavailable(NO_DATA_MSG)
        return clean_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))
    def quote(self, symbol): return QuoteInfo(symbol=symbol)
