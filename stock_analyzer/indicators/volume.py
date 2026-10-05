"""Volumenanalyse: relatives Volumen, OBV, Volumenprofil (für S/R-Zonen)."""
import numpy as np
import pandas as pd


def relative_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Volumen / Durchschnittsvolumen der VORHERIGEN n Bars (aktuelle Bar fließt nicht in ihren eigenen Vergleich)."""
    base = df["Volume"].rolling(n, min_periods=n).mean().shift(1)
    return (df["Volume"] / base).replace([np.inf, -np.inf], np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def volume_profile_levels(df: pd.DataFrame, bins: int = 40, top: int = 3) -> list[float]:
    """Preisniveaus mit dem höchsten gehandelten Volumen (grobes Volumenprofil auf Basis der Schlusskurse)."""
    if df["Volume"].sum() <= 0:
        return []
    hist, edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
    idx = np.argsort(hist)[::-1][:top]
    return [float((edges[i] + edges[i + 1]) / 2) for i in idx if hist[i] > 0]
