from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import scanner  # uses scan_one()

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
INDEX_HTML = WEB_DIR / "index.html"

app = FastAPI(title="Jifu 2h Scanner")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(INDEX_HTML))

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/api/scan")
def api_scan(tickers: str = Query(..., description="comma-separated: AAPL,MSFT")):
    syms = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    out = []
    for sym in syms:
        res = scanner.scan_one(sym)   # BUY-only logic + filters
        if res and "BUY" in res:
            b = res["BUY"]
            out.append({
                "ticker": sym,
                "price": b["price"],
                "time": b["time"],
                "dd_pct": b["drawdown_pct"],
                "macd": b["macd"],
                "mom6": b["mom6"],
                "mom12": b["mom12"],
            })
    out.sort(key=lambda x: -x["dd_pct"])
    return out
from typing import Optional
import scanner

@app.get("/api/scan_all")
def api_scan_all(limit: int = 2500, page_size: int = 200, page: int = 1):
    """
    Scan the US universe in pages to avoid timeouts.
    - limit: maximum tickers to consider
    - page_size: how many to scan in this call (keep 100–300 on free tier)
    - page: which page (1-based)
    """
    universe = scanner.list_us_tickers(limit=limit)
    # page math
    start = max(0, (page - 1) * page_size)
    end = min(len(universe), start + page_size)
    tickers_slice = universe[start:end]

    hits = scanner.scan_many(tickers_slice)
    return {
        "total": len(universe),
        "page": page,
        "page_size": page_size,
        "start": start,
        "end": end,
        "hits": hits,   # BUY signals only
    }
