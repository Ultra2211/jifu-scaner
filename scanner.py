# scanner.py
import os, sys, time, math, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ===== Settings via env =====
POLY_KEY      = os.getenv("POLYGON_API_KEY", "")
MAX_TICKERS   = int(os.getenv("MAX_TICKERS", "2500"))
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "220"))
MIN_PRICE     = float(os.getenv("MIN_PRICE", "3"))

# “deep fall” gates (you can leave them; we now also enforce 5 down days)
DD_PCT_MIN    = float(os.getenv("DD_PCT_MIN", "6"))
DEEP_BARS_MIN = int(os.getenv("DEEP_BARS_MIN", "3"))

# momentum gates (keep >0 by default)
MOM6_MIN   = float(os.getenv("MOM6_MIN", "0"))
MOM12_MIN  = float(os.getenv("MOM12_MIN", "0"))

USE_HA = os.getenv("USE_HEIKIN_ASHI", "false").lower() in ("1","true","yes","y")

if not POLY_KEY:
    print("Set POLYGON_API_KEY in env", file=sys.stderr); sys.exit(1)

BASE = "https://api.polygon.io"

def poly_get(url, params=None):
    params = (params or {}) | {"apiKey": POLY_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- Universe
def list_us_tickers(limit=MAX_TICKERS):
    url = f"{BASE}/v3/reference/tickers"
    params = {"market":"stocks","active":"true","type":"CS","limit":1000}
    out, next_url = [], url
    while next_url and len(out) < limit:
        data = poly_get(next_url, params if next_url.endswith("/tickers") else None)
        for t in data.get("results", []):
            if t.get("primary_exchange") in ("XNYS","XNAS","XASE","ARCX","BATS","IEXG"):
                out.append(t["ticker"])
        next_url = data.get("next_url")
    return sorted(set(out))[:limit]

# ---------- Data
def fetch_2h(symbol, bars=LOOKBACK_BARS):
    days = max(10, math.ceil(bars/12) + 5)
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    end   = datetime.now(timezone.utc).date().isoformat()
    url = f"{BASE}/v2/aggs/ticker/{symbol}/range/2/hour/{start}/{end}"
    data = poly_get(url, {"adjusted":"true","limit":50000})
    res = data.get("results", [])
    if not res: return None
    df = pd.DataFrame(res).rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True).tail(bars)

def fetch_daily(symbol, lookback_days=420):
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    end   = datetime.now(timezone.utc).date().isoformat()
    url = f"{BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    data = poly_get(url, {"adjusted":"true","limit":50000})
    res = data.get("results", [])
    if not res: return None
    df = pd.DataFrame(res).rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close"})
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)

# ---------- Indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def macd_line(c): return ema(c,12) - ema(c,26)

def momentum_pct(series, bars_back):
    if len(series) <= bars_back: return None
    old = float(series.iloc[-bars_back-1]); now = float(series.iloc[-1])
    if old <= 0: return None
    return (now - old) / old * 100.0

def to_heikin_ashi(df):
    ha = df.copy()
    o, h, l, c = ha["open"].values, ha["high"].values, ha["low"].values, ha["close"].values
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.zeros_like(ha_close); ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low  = np.minimum.reduce([l, ha_open, ha_close])
    ha["open"], ha["high"], ha["low"], ha["close"] = ha_open, ha_high, ha_low, ha_close
    return ha

# ---------- Helper: >=5 consecutive down DAILY closes ending before now
def has_five_consecutive_down_days(ddf):
    if ddf is None or len(ddf) < 7:  # safety
        return False
    # count consecutive days where close < prior close
    closes = ddf["close"].values
    down = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] < closes[i-1]:
            down += 1
        else:
            break
    return down >= 5

# ---------- BUY / SELL rules (2h)
def detect_buy(df2h):
    sig = to_heikin_ashi(df2h) if USE_HA else df2h.copy()
    sig["ema9"] = ema(sig["close"], 9)
    sig["macd"] = macd_line(sig["close"])
    sig["is_green"] = sig["close"] > sig["open"]

    # We require EMA9 to "pass the body" of a green candle and a classic cross
    body_low  = np.minimum(sig["open"], sig["close"])
    body_high = np.maximum(sig["open"], sig["close"])
    ema_inside_body = (sig["ema9"] >= body_low) & (sig["ema9"] <= body_high)

    prev_below_eq = sig["close"].shift(1) <= sig["ema9"].shift(1)
    now_above     = sig["close"] > sig["ema9"]

    # Optional: ensure a “deep fall” context (kept from earlier)
    hh = sig["high"].rolling(100, min_periods=2).max()
    sig["dd_pct"] = (hh - sig["close"]) / hh * 100.0
    below = (sig["close"] < sig["ema9"]).astype(int)
    run=0; streak=[]
    for b in below:
        run = run + 1 if b else 0
        streak.append(run)
    sig["below_streak"] = streak
    deep_prev = (sig["dd_pct"].shift(1) >= DD_PCT_MIN) & (sig["below_streak"].shift(1) >= DEEP_BARS_MIN)

    buy = (
        sig["is_green"] &
        prev_below_eq & now_above &
        ema_inside_body &                  # EMA9 must pass through body (not only wick)
        (sig["macd"] <= 0) &
        deep_prev
    )
    return buy, sig

def detect_sell(df2h):
    sig = to_heikin_ashi(df2h) if USE_HA else df2h.copy()
    sig["ema21"] = ema(sig["close"], 21)     # BMSB green line approximation
    sig["is_red"] = sig["close"] < sig["open"]

    # “Green line passes a red candle body” (not tail)
    body_low  = np.minimum(sig["open"], sig["close"])
    body_high = np.maximum(sig["open"], sig["close"])
    ema_in_body = (sig["ema21"] >= body_low) & (sig["ema21"] <= body_high)

    # Optional cross direction (from above to below):
    prev_above = sig["close"].shift(1) >= sig["ema21"].shift(1)
    now_below  = sig["close"] <= sig["ema21"]

    sell = sig["is_red"] & ema_in_body & prev_above & now_below
    return sell, sig

# ---------- Scan one symbol
def scan_one(sym):
    try:
        ddf = fetch_daily(sym)
        if ddf is None or len(ddf) < 260:
            return None

        # price gate
        last_close = float(ddf["close"].iloc[-1])
        if last_close < MIN_PRICE:
            return None

        # momentum gates
        m6  = momentum_pct(ddf["close"], 126)
        m12 = momentum_pct(ddf["close"], 252)
        if m6 is None or m12 is None: return None
        if not (m6 > MOM6_MIN and m12 > MOM12_MIN): return None

        # “≥5 down days” (daily) requirement
        if not has_five_consecutive_down_days(ddf):
            return None

        df2h = fetch_2h(sym)
        if df2h is None or len(df2h) < 60:
            return None

        buy_mask, bdf = detect_buy(df2h)
        sell_mask, sdf = detect_sell(df2h)

        out = {}
        if bool(buy_mask.iloc[-1]):
            last = bdf.iloc[-1]
            out["BUY"] = {
                "time": str(last["time"]),
                "price": float(last["close"]),
                "macd": float(round((bdf["macd"].iloc[-1] if "macd" in bdf else 0.0), 5)),
            }
        if bool(sell_mask.iloc[-1]):
            last = sdf.iloc[-1]
            out["SELL"] = {
                "time": str(last["time"]),
                "price": float(last["close"]),
            }
        return out if out else None
    except Exception:
        return None

# ---------- Batch scan with polite pacing
def scan_many(symbols):
    hits = []
    for sym in symbols:
        res = scan_one(sym)
        if not res: 
            time.sleep(0.10); 
            continue
        item = {"ticker": sym}
        if "BUY" in res:
            item["BUY"] = res["BUY"]
        if "SELL" in res:
            item["SELL"] = res["SELL"]
        hits.append(item)
        time.sleep(0.10)
    return hits

