# BSCGPT – Black–Scholes & GPT Analysis for Stocks

This script combines **historical price data**, a **Black–Scholes option model**, and an optional **GPT-based qualitative analysis** into a single tool.

It:

- Pulls historical daily bars for a given stock from **Alpaca**
- Estimates **historical volatility**
- Builds a **Black–Scholes snapshot** (call/put prices + Greeks)
- Optionally calls **OpenAI** to generate a compact, trader-focused commentary
- Plots the **closing price history** for the selected period

---

## What the script does (step-by-step)

Given a ticker and period (from environment variables):

1. **Reads config from environment**  
   - `SYMBOL` – stock ticker to analyze  
   - `PERIOD` – how much history to pull (e.g. `6mo`, `1y`, `2y`, `ytd`)  
   - `ALPACA_DATA_FEED` – `iex` or `sip` (defaults to `iex`)  

2. **Fetches historical bars from Alpaca**  
   Using `StockHistoricalDataClient` and `StockBarsRequest`, it loads daily candles (`Open`, `High`, `Low`, `Close`, `Volume`) into a `pandas` DataFrame.

3. **Computes historical volatility**  
   - Uses log returns of `Close`
   - Annualises with `TRADING_DAYS = 252`
   - If the computed volatility is invalid, falls back to `σ = 0.20`

4. **Builds a Black–Scholes snapshot**  
   Using the `BlackScholesModel` class, it computes:
   - Prices:
     - Call price
     - Put price
   - Greeks:
     - Call Delta / Put Delta  
     - Gamma  
     - Theta (per day, call & put)  
     - Vega (per 1% vol change)  
     - Rho (per 1% rate change, call & put)

   Inputs:
   - `S` – current price (last close)
   - `K` – strike (rounded current price)
   - `T` – time to expiration in years (default `DEFAULT_T_DAYS = 30` days)
   - `r` – risk-free rate (`RISK_FREE_RATE = 0.035`)
   - `σ` – historical vol as above

5. **Optional GPT analysis**  
   If enabled via env vars, the script calls OpenAI and asks for:
   - A concise, bullet-point assessment
   - Commentary on directional bias, risk, Greeks, and P&L impact under spot/vol/time shocks
   - A tiny 3-row “Greek | Meaning | Scenario” table

6. **Plots price history**  
   Shows a Matplotlib chart of the `Close` price for the chosen period.

---

## Data Sources

- **Market data:**  
  [Alpaca Market Data](https://alpaca.markets/) via `alpaca-py` (`StockHistoricalDataClient`, `DataFeed.IEX` or `DataFeed.SIP`).

- **GPT analysis (optional):**  
  [OpenAI Chat Completions API](https://platform.openai.com/) through the official `openai` Python client.

---

## Requirements

Python 3.10+ is recommended.

Core dependencies (put these in `requirements.txt` if you haven’t already):

```txt
alpaca-py
pandas
numpy
matplotlib
scipy
python-dotenv
openai
