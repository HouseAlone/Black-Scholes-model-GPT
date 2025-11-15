import os
import re
from datetime import datetime, timedelta, UTC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Alpaca Market Data ---
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# -------------- Config --------------
DEFAULT_T_DAYS = 30   # days to expiration used for BS model if no options data
RISK_FREE_RATE = 0.035
TRADING_DAYS = 252


# -------------- GPT env helpers --------------
def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def get_openai_model() -> str:
    return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()


# -------------- Env readers --------------
def get_env_ticker() -> str:
    val = os.getenv("SYMBOL")
    if not val:
        raise EnvironmentError("SYMBOL not set in .env. Add SYMBOL=YourTicker (e.g., SYMBOL=AAPL)")
    val = val.strip().upper()
    if not re.fullmatch(r"^[A-Z0-9]{1,10}([.-][A-Z0-9]{1,10})?$", val):
        raise ValueError(f"Invalid SYMBOL in env: {val}")
    return val


def get_env_period() -> str:
    val = os.getenv("PERIOD") or os.getenv("TIMEFRAME")
    if not val:
        raise EnvironmentError("PERIOD not set in .env. Add PERIOD=10y (or 6mo, 1y, ytd, max, etc.)")
    val = val.strip().lower()
    if not re.fullmatch(r"^(?:\d+(?:d|mo|y)|ytd|max)$", val):
        raise ValueError(f"Invalid PERIOD in env: {val}. Expected like 1d, 5d, 1mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    return val


def get_env_data_feed() -> str:
    val = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()
    if val not in {"iex", "sip"}:
        raise ValueError("ALPACA_DATA_FEED must be 'iex' or 'sip'")
    return val


# -------------- Period translation --------------
def period_to_dates(period_str: str) -> tuple[datetime, datetime]:
    today = datetime.now(UTC)
    if period_str == "ytd":
        start = datetime(today.year, 1, 1, tzinfo=UTC)
        return start, today
    if period_str == "max":
        start = today - timedelta(days=365*15)
        return start, today

    m = re.fullmatch(r"(\d+)(d|mo|y)", period_str)
    assert m is not None
    n = int(m.group(1))
    unit = m.group(2)
    if unit == 'd':
        start = today - timedelta(days=n)
    elif unit == 'mo':
        start = today - timedelta(days=30*n)
    else:  # 'y'
        start = today - timedelta(days=365*n)
    return start, today


# -------------- Alpaca price fetch --------------
def get_alpaca_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")
    return StockHistoricalDataClient(api_key=key, secret_key=secret)


def fetch_price_history_alpaca(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = get_alpaca_client()
    feed_str = get_env_data_feed()
    feed_enum = DataFeed.IEX if feed_str == "iex" else DataFeed.SIP
    # Avoid “recent data” restrictions by trimming the end time back ~20 min
    safe_end = min(end, datetime.now(UTC) - timedelta(minutes=20))

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start,
        end=safe_end,
        limit=None,
        feed=feed_enum,
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df is None or df.empty:
        raise ValueError(f"Alpaca returned no bars for {ticker} in the selected range.")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level=0)
    out = df[[
        next(c for c in df.columns if c.lower() == 'open'),
        next(c for c in df.columns if c.lower() == 'high'),
        next(c for c in df.columns if c.lower() == 'low'),
        next(c for c in df.columns if c.lower() == 'close'),
        next(c for c in df.columns if c.lower() == 'volume'),
    ]].copy()
    out.index.name = 'timestamp'
    out.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return out


# -------------- Black–Scholes --------------
class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        d1, d2 = self._d1(), self._d2()
        return self.S * si.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)

    def put_price(self):
        d1, d2 = self._d1(), self._d2()
        return self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2) - self.S * si.norm.cdf(-d1)

    def call_delta(self):
        return si.norm.cdf(self._d1())

    def put_delta(self):
        return si.norm.cdf(self._d1()) - 1.0

    def gamma(self):
        return si.norm.pdf(self._d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * np.sqrt(self.T) * si.norm.pdf(self._d1()) / 100.0

    def call_theta(self):
        d1, d2 = self._d1(), self._d2()
        term1 = -(self.S * si.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)
        return (term1 + term2) / 365.0

    def put_theta(self):
        d1, d2 = self._d1(), self._d2()
        term1 = -(self.S * si.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2)
        return (term1 + term2) / 365.0

    def call_rho(self):
        d2 = self._d2()
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(d2) / 100.0

    def put_rho(self):
        d2 = self._d2()
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-d2) / 100.0


# -------------- Utilities --------------
def annualized_hist_vol(close: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    log_ret = np.log(close / close.shift(1)).dropna()
    return float(np.sqrt(trading_days) * log_ret.std())


# -------------- GPT Analysis --------------
def gpt_analysis(ticker: str,
                 period: str,
                 hist: pd.DataFrame,
                 S: float,
                 K: float,
                 T: float,
                 r: float,
                 sigma: float,
                 call_price: float,
                 put_price: float,
                 greeks: dict) -> None:
#GPT Analysis

    if not get_env_bool("ANALYZE_WITH_GPT", True):
        print("\n[GPT] Skipping analysis (ANALYZE_WITH_GPT is false).")
        return
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[GPT] Skipping analysis: OPENAI_API_KEY not set in environment.")
        return
    try:
        client = OpenAI(api_key=api_key)
        model = get_openai_model()

        # Summarize series stats to keep prompt compact
        close = hist["Close"].dropna()
        n = int(close.shape[0])
        ret = np.log(close / close.shift(1)).dropna()
        ann_vol = float(np.sqrt(TRADING_DAYS) * ret.std()) if not ret.empty else float("nan")
        change_1m = float((close.iloc[-1] / close.iloc[max(0, n-21)] - 1) * 100) if n > 21 else float("nan")
        change_3m = float((close.iloc[-1] / close.iloc[max(0, n-63)] - 1) * 100) if n > 63 else float("nan")
        change_1y = float((close.iloc[-1] / close.iloc[max(0, n-252)] - 1) * 100) if n > 252 else float("nan")

        g_text = "\n".join([f"- {k}: {v:.6f}" for k, v in greeks.items()])

        prompt = (
            "You are a quantitative trading assistant. Given a stock's recent behavior and a Black–Scholes snapshot, "
            "write a concise assessment for an informed retail trader. Discuss directional bias, risk, and how Greeks "
            "map to P&L under shocks (spot up/down, vol up/down, time decay). The information you provide will be used "
            "as an extra piece of information of an investment group, though not influencing decision making directly"
            " Use bullet points. Keep it under 300 words.\n\n"
            f"Ticker: {ticker}\n"
            f"Period: {period}\n"
            f"Observations (computed):\n"
            f"- Samples: {n}\n- 1M change (%): {change_1m:.2f}\n- 3M change (%): {change_3m:.2f}\n- 1Y change (%): {change_1y:.2f}\n- Ann. vol (hist): {ann_vol:.4f}\n\n"
            f"BS inputs:\n- S: {S:.4f}\n- K: {K}\n- T (years): {T:.6f}\n- r: {r:.4f}\n- sigma: {sigma:.4f}\n"
            f"Prices:\n- Call: {call_price:.4f}\n- Put: {put_price:.4f}\n\n"
            f"Greeks:\n{g_text}\n\n"
            "Return also a tiny 3-row table like: 'Greek | Meaning | If spot +2%, vol +5%, 7 days pass'."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        print("\n--- GPT Analysis ---")
        print(resp.choices[0].message.content.strip())
    except Exception as e:
        print("\n[GPT] Analysis failed:", str(e))


# -------------- Main flow --------------
def main():
    ticker = get_env_ticker()
    period = get_env_period()
    start, end = period_to_dates(period)

    print(f"Fetching {ticker} bars from Alpaca: {start.date()} → {end.date()} ...")
    print(f"Data feed: {get_env_data_feed().upper()}")
    hist = fetch_price_history_alpaca(ticker, start, end)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker} in selected range.")

    current_price = float(hist['Close'].iloc[-1])
    print(f"Current Stock Price (S): {current_price:.4f}")

    hist_vol = annualized_hist_vol(hist['Close'])
    sigma = hist_vol if np.isfinite(hist_vol) and hist_vol > 0 else 0.2
    print(f"Historical Volatility (σ): {hist_vol:.4f}  | Using σ for model: {sigma:.4f}")

    T = DEFAULT_T_DAYS / 365.0
    print(f"Time to Expiration (T): {T:.6f} years (default {DEFAULT_T_DAYS} days)")

    K = round(current_price)
    print(f"Strike price (K): {K}")

    r = RISK_FREE_RATE
    print(f"Risk-Free Rate (r): {r}")

    bsm = BlackScholesModel(S=current_price, K=K, T=T, r=r, sigma=sigma)

    call_price = bsm.call_price()
    put_price = bsm.put_price()

    greeks = {
        "Delta (Call)": bsm.call_delta(),
        "Delta (Put)": bsm.put_delta(),
        "Gamma": bsm.gamma(),
        "Theta (Call/day)": bsm.call_theta(),
        "Theta (Put/day)": bsm.put_theta(),
        "Vega (/1% vol)": bsm.vega(),
        "Rho (Call/1%)": bsm.call_rho(),
        "Rho (Put/1%)": bsm.put_rho(),
    }

    print("\nPrices:")
    print(f"  Call: {call_price:.4f}")
    print(f"  Put : {put_price:.4f}")

    print("\nGreeks:")
    for k, v in greeks.items():
        print(f"  {k}: {v:.6f}")

    gpt_analysis(ticker, period, hist, current_price, K, T, r, sigma, call_price, put_price, greeks)

    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist['Close'], label=ticker)
    plt.title(f"{ticker} — Close (Alpaca)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()