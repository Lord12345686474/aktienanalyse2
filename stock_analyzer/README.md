# Technischer Aktienanalyse-Assistent

Start (aus dem Ordner über `stock_analyzer/`):

    pip install -r stock_analyzer/requirements.txt
    streamlit run stock_analyzer/ui/dashboard.py        # Oberfläche
    python -m stock_analyzer.main AAPL --multi           # Kommandozeile
    python -m unittest stock_analyzer.tests.test_all     # Tests

## Architektur
- data/ – austauschbarer Datenanbieter (`MarketDataProvider`; yfinance, CSV). Fehlende Daten -> `DataUnavailable`, nie erfundene Werte.
- indicators/ – RSI (Wilder, exakt), MACD, SMA/EMA, ATR, Volumen/OBV/Volumenprofil.
- patterns/ – 12 Candlestick-Formationen (kausal).
- analysis/ – Pivots + Divergenzen, Trend/Ereignisse, S/R-Zonen, Scoring je Komponente, Signal-Engine, Multi-Timeframe.
- backtesting/ – Ausführung zum Open der Folgekerze, Gebühren, keine Zukunftsdaten.
- ui/ – Streamlit + Plotly.

## Entscheidungslogik (kurz)
Gewichteter Score (-100..+100). KAUFEN/VERKAUFEN nur, wenn Score über Schwelle, >= 3 Komponenten übereinstimmen,
RSI und MACD nicht widersprechen und – gegen einen intakten Haupttrend – ein bestätigtes Wendesignal vorliegt. Sonst ABWARTEN.
"Signalstärke" ist heuristisch und KEINE Wahrscheinlichkeit. Schwellen/Gewichte sind unkalibriert. Keine Anlageberatung.
