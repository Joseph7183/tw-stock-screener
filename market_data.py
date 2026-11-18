"""資料來源：
- 證交所／櫃買中心 OpenAPI：全市場當日收盤價、成交值、本益比（免費、免金鑰）
- FinMind：個股多年財報、月營收、股價歷史（需免費 token）
"""
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from config import FINMIND_SLEEP, FINMIND_URL

TWSE_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_PE_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_DAY_ALL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_PE_ALL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"


def _to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _first(row, *keys):
    """不同 API 的欄位名稱不一致，依序找第一個存在的。"""
    for k in keys:
        if k in row and row[k] not in (None, "", "--", "-"):
            return row[k]
    return None


def _get_json(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def market_snapshot():
    """全市場當日快照：DataFrame[stock_id, name, market, close, trade_value, pe]。"""
    rows = {}
    for row in _get_json(TWSE_DAY_ALL):
        sid = _first(row, "Code")
        if sid:
            rows[sid] = {"stock_id": sid, "name": _first(row, "Name"), "market": "TWSE",
                         "close": _to_float(_first(row, "ClosingPrice")),
                         "change": _to_float(_first(row, "Change")),
                         "trade_value": _to_float(_first(row, "TradeValue")), "pe": None, "div_yield": None}
    for row in _get_json(TWSE_PE_ALL):
        sid = _first(row, "Code")
        if sid in rows:
            rows[sid]["pe"] = _to_float(_first(row, "PEratio"))
            dy = _to_float(_first(row, "DividendYield"))
            rows[sid]["div_yield"] = dy / 100 if dy is not None else None
    for row in _get_json(TPEX_DAY_ALL):
        sid = _first(row, "SecuritiesCompanyCode", "Code")
        if sid:
            rows[sid] = {"stock_id": sid, "name": _first(row, "CompanyName", "Name"), "market": "TPEX",
                         "close": _to_float(_first(row, "Close", "ClosingPrice")),
                         "change": _to_float(_first(row, "Change")),
                         "trade_value": _to_float(_first(row, "TransactionAmount", "TradeValue")), "pe": None, "div_yield": None}
    for row in _get_json(TPEX_PE_ALL):
        sid = _first(row, "SecuritiesCompanyCode", "Code")
        if sid in rows:
            rows[sid]["pe"] = _to_float(_first(row, "PriceEarningRatio", "PEratio"))
            dy = _to_float(_first(row, "YieldRatio", "DividendYield"))
            rows[sid]["div_yield"] = dy / 100 if dy is not None else None
    df = pd.DataFrame(rows.values())
    if df.empty or df["close"].isna().all():
        raise RuntimeError("證交所／櫃買中心資料抓不到或欄位改名了，請檢查 market_data.py 的欄位名稱")
    return df


# ── FinMind ─────────────────────────────────────────────
_last_call = 0.0


def finmind(dataset, data_id=None, years=7):
    """呼叫 FinMind，自動限速。回傳 DataFrame。"""
    global _last_call
    wait = FINMIND_SLEEP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    params = {"dataset": dataset}
    if years:
        params["start_date"] = (date.today() - timedelta(days=int(365 * years))).isoformat()
    if data_id:
        params["data_id"] = data_id
    headers = {"Authorization": f"Bearer {os.environ['FINMIND_TOKEN']}"}
    for attempt in range(3):
        _last_call = time.time()
        r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=60)
        if r.status_code == 402 or "upper limit" in r.text:  # 額度用完，等一小時
            print("FinMind 額度用完，暫停 61 分鐘…")
            time.sleep(61 * 60)
            continue
        r.raise_for_status()
        return pd.DataFrame(r.json().get("data", []))
    raise RuntimeError(f"FinMind {dataset} {data_id} 連續失敗")


def stock_info():
    """FinMind 股票清單（含產業別）。"""
    df = finmind("TaiwanStockInfo", years=None)
    return df.drop_duplicates("stock_id", keep="last").set_index("stock_id")
