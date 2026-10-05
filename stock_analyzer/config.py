"""Zentrale, vom Benutzer veränderbare Einstellungen (alles in einer Dataclass)."""
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- Indikator-Parameter (klassische Defaults) ---
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_periods: tuple = (20, 50, 200)
    ema_periods: tuple = (20, 50)
    volume_ma: int = 20
    atr_period: int = 14

    # --- Gewichtung der Komponenten (wird automatisch normalisiert) ---
    weights: dict = field(default_factory=lambda: {
        "RSI": 25, "MACD": 30, "Candlesticks": 20, "Trend": 10,
        "Unterstützung/Widerstand": 5, "Volumen": 5, "Gleitende Durchschnitte": 5,
    })

    # --- Entscheidungsschwellen (Score -100..+100) ---
    buy_threshold: float = 25.0     # unkalibriert – per Backtest prüfen und anpassen
    sell_threshold: float = -25.0
    min_agreeing_components: int = 3     # mind. so viele Komponenten müssen in die gleiche Richtung zeigen

    # --- Pivot-/Struktur-Erkennung ---
    pivot_window: int = 5                # Bars links/rechts für Swing-Hoch/-Tief (Bestätigung erst nach n Bars!)
    divergence_max_age: int = 30         # Divergenz zählt nur, wenn das 2. Pivot höchstens so alt ist

    # --- Multi-Timeframe ---
    horizon: str = "long"                # "long" (Tag/Woche stark) oder "short" (kleine TF stärker)

    def normalized_weights(self) -> dict:
        total = sum(max(v, 0) for v in self.weights.values()) or 1.0
        return {k: max(v, 0) / total for k, v in self.weights.items()}
