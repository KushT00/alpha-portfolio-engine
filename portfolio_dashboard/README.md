# NIFTY Portfolio Analytics & Greeks Dashboard

A Flask-based web application to monitor, analyze, and visualize your option trading portfolio metrics, Black-Scholes Greeks (Delta, Theta, Gamma, Vega), market sentiment bias, and expiry distribution in real-time.

## Features
- **Live OpenAlgo Integration**: Automatically connects to OpenAlgo (`http://127.0.0.1:5000`) for positionbook and quotes.
- **Black-Scholes Option Greeks Engine**: Pure Python calculation of Delta ($\Delta$), Theta ($\Theta$), Gamma ($\Gamma$), and Vega ($V$).
- **Portfolio Sentiment Analysis**: Categorizes net directional bias (Strongly Bullish, Delta Neutral, Strongly Bearish) and calculates stress test scenarios ($\pm 100$-point NIFTY move).
- **Interactive Expiry Chart**: Chart.js bar chart displaying Call vs Put quantities/lots across option expiries.
- **Modern Dark Glassmorphism UI**: Real-time auto-refresh countdown, Min Premium filter, and detailed active position legs table.

## Quick Start

1. Ensure your OpenAlgo platform server is running on `http://127.0.0.1:5000`.
2. Run the dashboard Flask server:
   ```bash
   python portfolio_dashboard/app.py
   ```
3. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:5050
   ```

## Configuration

You can customize the OpenAlgo connection via environment variables:
- `OPENALGO_HOST`: Default is `http://127.0.0.1:5000`
- `OPENALGO_API_KEY`: Your OpenAlgo API key
- `PORT`: Dashboard web port (Default: `5050`)
