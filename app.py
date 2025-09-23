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
