from typing import Optional
import scanner

@app.get("/api/scan_all")
def api_scan_all(
    limit: int = 2500,
    page_size: int = 200,
    page: int = 1,
    side: str = "both"  # "buy", "sell", "both"
):
    universe = scanner.list_us_tickers(limit=limit)
    start = max(0, (page - 1) * page_size)
    end = min(len(universe), start + page_size)
    tickers_slice = universe[start:end]

    raw_hits = scanner.scan_many(tickers_slice)

    # filter side if requested
    def keep(item):
        if side == "buy":  return "BUY"  in item
        if side == "sell": return "SELL" in item
        return ("BUY" in item) or ("SELL" in item)

    hits = [h for h in raw_hits if keep(h)]
    return {
        "total": len(universe),
        "page": page,
        "page_size": page_size,
        "start": start,
        "end": end,
        "hits": hits,
    }
