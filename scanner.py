import os, sys, time, math, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

POLY_KEY    = os.getenv("POLYGON_API_KEY", "")
UNIVERSE    = os.getenv("UNIVERSE", "US").upper()
CUSTOM      = [s.strip().upper() for s in os.getenv("CUSTOM_SYMBOLS","").split(",") if s.strip()]
DD_PCT_MIN    = float(os.getenv("DD_PCT_MIN", "8"))
DEEP_BARS_MIN = int(os.getenv("DEEP_BARS_MIN", "5"))
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "220"))
MAX_TICKERS   = int(os.getenv("MAX_TICKERS", "2500"))
MOM6_MIN   = float(os.getenv("MOM6_MIN", "0"))
MOM12_MIN  = float(os.getenv("MOM12_MIN", "0"))
MIN_PRICE  = float(os.getenv("MIN_PRICE", "3"))
USE_HA     = os.getenv("USE_HEIKIN_ASHI", "false").lower() in ("1","true","yes","y")

if not POLY_KEY:
    print("Set POLYGON_API_KEY in env", file=sys.stderr); sys.exit(1)

BASE = "https://api.polygon.io"

def poly_get(url, params=None):
    params = (params or {}) | {"apiKey": POLY_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_2h(symbol, bars=LOOKBACK_BARS):
    days = max(10, math.ceil(bars/12)+5)
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
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low  = np.minimum.reduce([l, ha_open, ha_close])
    ha["open"], ha["high"], ha["low"], ha["close"] = ha_open, ha_high, ha_low, ha_close
    return ha

def detect_buy(df):
    sig = to_heikin_ashi(df) if USE_HA else df.copy()
    sig["ema9"] = ema(sig["close"], 9)
    sig["macd"] = macd_line(sig["close"])
    sig["is_green"] = sig["close"] > sig["open"]

    hh = sig["high"].rolling(100, min_periods=2).max()
    sig["dd_pct"] = (hh - sig["close"]) / hh * 100.0

    below = (sig["close"] < sig["ema9"]).astype(int)
    streak = []; run=0
    for b in below:
        run = run + 1 if b else 0
        streak.append(run)
    sig["below_streak"] = streak

    prev_le = sig["close"].shift(1) <= sig["ema9"].shift(1)
    now_gt  = sig["close"] > sig["ema9"]

    body_low  = np.minimum(sig["open"], sig["close"])
    body_high = np.maximum(sig["open"], sig["close"])
    ema_inside_body = (sig["ema9"] >= body_low) & (sig["ema9"] <= body_high)

    deep_prev = (sig["dd_pct"].shift(1) >= DD_PCT_MIN) & (sig["below_streak"].shift(1) >= DEEP_BARS_MIN)

    buy = deep_prev & sig["is_green"] & prev_le & now_gt & ema_inside_body & (sig["macd"] < 0)
    return buy, sig["dd_pct"], sig

def scan_one(sym):
    try:
        ddf = fetch_daily(sym)
        if ddf is None or len(ddf) < 260: return None
        last_close = float(ddf["close"].iloc[-1])
        if last_close < MIN_PRICE: return None

        m6  = momentum_pct(ddf["close"], 126)
        m12 = momentum_pct(ddf["close"], 252)
        if m6 is None or m12 is None: return None
        if not (m6 > MOM6_MIN and m12 > MOM12_MIN): return None

        df = fetch_2h(sym)
        if df is None or len(df) < 60: return None
        buy, dd_pct, sig = detect_buy(df)
        if not buy.iloc[-1]: return None

        last = sig.iloc[-1]
        return {
            "BUY": {
                "time": str(last["time"]),
                "price": float(last["close"]),
                "drawdown_pct": float(round(dd_pct.iloc[-1], 2)),
                "macd": float(round(sig["macd"].iloc[-1], 5)),
                "mom6": round(m6, 2),
                "mom12": round(m12, 2),
            }
        }
    except Exception:
        return None
        # ---------- Universe listing (US common stocks on major exchanges)
def list_us_tickers(limit=2500):
    url = f"{BASE}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": 1000,
        # You can add 'ticker.gte' or 'ticker.lt' for paging alphabetically if you want
    }
    out = []
    next_url = url
    while next_url and len(out) < limit:
        data = poly_get(next_url, params if next_url.endswith("/tickers") else None)
        for t in data.get("results", []):
            if t.get("primary_exchange") in ("XNYS", "XNAS", "XASE", "ARCX", "BATS", "IEXG"):
                out.append(t["ticker"])
        next_url = data.get("next_url")
    return sorted(set(out))[:limit]


# ---------- Batch scan helper (returns BUY hits only)
def scan_many(symbols):
    hits = []
    for i, sym in enumerate(symbols, 1):
        res = scan_one(sym)
        if res and "BUY" in res:
            b = res["BUY"]
            hits.append({
                "ticker": sym,
                "price": b["price"],
                "time": b["time"],
                "dd_pct": b["drawdown_pct"],
                "macd": b["macd"],
                "mom6": b["mom6"],
                "mom12": b["mom12"],
            })
        # polite pacing: Polygon Starter is fine but this avoids spikes
        time.sleep(0.10)
    # rank by drawdown (largest first)
    hits.sort(key=lambda x: -x["dd_pct"])
    return hits

