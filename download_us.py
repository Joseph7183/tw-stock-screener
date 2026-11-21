"""下載重建 009820（那斯達克 100 增強成長指數）回測需要的美股資料。可續傳。

存放位置 data/us/：
- ndx_members.csv           每季末的那斯達克 100 成分股（point-in-time，含已被剔除的公司）
- prices/<TICKER>.csv.gz    日股價（Yahoo，含 Adj Close＝含息還原價）
- facts/<TICKER>.csv.gz     SEC XBRL 季報科目（含 filed 申報日，回測只用已申報的數字）
- _missing.json             抓不到的代號，回測時列為限制

用法：python download_us.py
需要套件：yfinance、nasdaq-100-ticker-history、requests、pandas
"""
import json
import os
import time
from datetime import date

import pandas as pd
import requests

OUT = "data/us"
UA = {"User-Agent": "tw-stock-screener research joseph7183@users.noreply.github.com"}
START = "2012-01-01"            # 規則要 3 年財報，回測從 2016 起
BENCH = ["QQQ"]                 # 那斯達克 100 本身（含息）

# XBRL 科目：(我們的名稱, [候選科目，依序嘗試])
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "Revenue"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfSales"],
    "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseIncludingAcquiredInProcessCost"],
    "shares": ["EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
               "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
}


def members():
    from nasdaq_100_ticker_history import tickers_as_of
    rows = []
    for y in range(2015, date.today().year + 1):
        for m, d in ((2, 28), (5, 31), (8, 31), (11, 30)):
            if date(y, m, d) > date.today():
                continue
            try:
                ts = tickers_as_of(y, m, d)
            except Exception as e:      # 套件涵蓋範圍外
                print(f"  {y}-{m}-{d} 成分股抓不到：{e}")
                continue
            rows += [{"date": f"{y}-{m:02d}-{d:02d}", "ticker": t} for t in sorted(ts)]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ndx_members.csv", index=False)
    print(f"成分股：{df['date'].nunique()} 個時點，共 {df['ticker'].nunique()} 檔曾經入選")
    return sorted(df["ticker"].unique())


def cik_map():
    j = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in j.values()}


def cik_lookup(ticker):
    """舊代號（已下市、被併購）不在最新對照表，改用 EDGAR 查詢頁。"""
    r = requests.get("https://www.sec.gov/cgi-bin/browse-edgar",
                     params={"action": "getcompany", "CIK": ticker, "type": "10-K", "count": "1", "output": "atom"},
                     headers=UA, timeout=60)
    import re
    m = re.search(r"<cik>(\d+)</cik>", r.text)
    return m.group(1).zfill(10) if m else None


def facts(ticker, cik):
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=UA, timeout=120)
    if r.status_code != 200:
        return pd.DataFrame()
    j = r.json().get("facts", {})
    rows = []
    for name, cands in CONCEPTS.items():
        for tax in ("us-gaap", "dei", "ifrs-full"):
            for c in cands:
                node = j.get(tax, {}).get(c)
                if not node:
                    continue
                for unit, vals in node["units"].items():
                    for v in vals:
                        rows.append({"item": name, "concept": f"{tax}:{c}", "unit": unit,
                                     "start": v.get("start"), "end": v.get("end"), "val": v.get("val"),
                                     "fy": v.get("fy"), "fp": v.get("fp"), "form": v.get("form"),
                                     "filed": v.get("filed")})
    return pd.DataFrame(rows)


def prices(ticker):
    import yfinance as yf
    df = yf.download(ticker, start=START, auto_adjust=False, actions=True, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()


def main():
    os.makedirs(f"{OUT}/prices", exist_ok=True)
    os.makedirs(f"{OUT}/facts", exist_ok=True)
    tickers = members()
    cmap = cik_map()
    missing = {"prices": [], "facts": []}
    for i, t in enumerate(tickers + BENCH, 1):
        yt = t.replace(".", "-")
        p = f"{OUT}/prices/{t}.csv.gz"
        if not os.path.exists(p):
            df = prices(yt)
            if df.empty:
                missing["prices"].append(t)
            else:
                df.to_csv(p, index=False)
            time.sleep(0.5)
        if t in BENCH:
            continue
        f = f"{OUT}/facts/{t}.csv.gz"
        if not os.path.exists(f):
            cik = cmap.get(t.upper()) or cik_lookup(t)
            time.sleep(0.2)
            df = facts(t, cik) if cik else pd.DataFrame()
            if df.empty:
                missing["facts"].append(t)
            else:
                df.assign(cik=cik).to_csv(f, index=False)
            time.sleep(0.2)
        if i % 25 == 0:
            print(f"  進度 {i}/{len(tickers) + len(BENCH)}")
    with open(f"{OUT}/_missing.json", "w") as fh:
        json.dump(missing, fh, indent=1)
    print(f"完成。股價抓不到 {len(missing['prices'])} 檔：{missing['prices']}")
    print(f"財報抓不到 {len(missing['facts'])} 檔：{missing['facts']}")


if __name__ == "__main__":
    main()
