"""回測用的歷史資料下載。可續傳：已經抓過的檔案會跳過，所以可以分很多次慢慢跑。

存放位置：data/<資料集>/<股票代號>.csv.gz
進度紀錄：data/_progress.json

    python download_data.py                  # 一直抓到時間用完
    python download_data.py --limit 3        # 只抓 3 檔（測試用）
    python download_data.py --minutes 300    # 最多跑 300 分鐘
"""
import argparse
import json
import os
import time
from datetime import date

import pandas as pd

from config import EXCLUDED_INDUSTRIES
from market_data import finmind

START = "2010-01-01"   # 回測從 2016 年開始，往前多抓 5 年給關卡用
DATA = "data"

# 每檔股票要抓的資料集
PER_STOCK = [
    "TaiwanStockFinancialStatements",   # 損益表（單季）
    "TaiwanStockBalanceSheet",          # 資產負債表
    "TaiwanStockCashFlowsStatement",    # 現金流量表（年初累計）
    "TaiwanStockPrice",                 # 日股價與成交值
    "TaiwanStockPER",                   # 每日本益比、殖利率
    "TaiwanStockMonthRevenue",          # 月營收
    "TaiwanStockDividend",              # 股利（算含息報酬）
]
BENCHMARKS = ["0050"]


def fetch(dataset, sid):
    years = (date.today() - date.fromisoformat(START)).days / 365
    df = finmind(dataset, sid, years=years)
    if df.empty:
        return df
    if "type" in df:  # 財報裡「佔總資產百分比」的列用不到，省空間
        df = df[~df["type"].astype(str).str.endswith("_per")]
    return df


def universe():
    """一般股（含已下市），排除 ETF、權證、金融業。"""
    info = finmind("TaiwanStockInfo", years=None)
    info = info.drop_duplicates("stock_id", keep="last")
    try:
        gone = finmind("TaiwanStockDelisting", years=None)
        gone = gone[["stock_id", "stock_name"]].assign(industry_category="已下市")
        print(f"已下市公司 {len(gone)} 檔")
    except Exception as e:  # 拿不到就先記下來，不中斷
        print(f"⚠ 下市清單抓不到：{e}")
        gone = pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])
    allx = pd.concat([info[["stock_id", "stock_name", "industry_category", "type"]], gone])
    allx = allx.drop_duplicates("stock_id", keep="first")
    allx = allx[allx["stock_id"].astype(str).str.fullmatch(r"[1-9]\d{3}")]
    allx = allx[~allx["industry_category"].isin(EXCLUDED_INDUSTRIES)]
    return allx.sort_values("stock_id")


def main(limit=None, minutes=330):
    deadline = time.time() + minutes * 60
    os.makedirs(DATA, exist_ok=True)
    uni_path = f"{DATA}/universe.csv"
    if os.path.exists(uni_path):
        uni = pd.read_csv(uni_path, dtype=str)
    else:
        uni = universe()
        uni.to_csv(uni_path, index=False)
    ids = BENCHMARKS + uni["stock_id"].tolist()
    if limit:
        ids = ids[:limit + len(BENCHMARKS)]

    todo = [(ds, sid) for sid in ids for ds in PER_STOCK
            if not os.path.exists(f"{DATA}/{ds}/{sid}.csv.gz")]
    total = len(ids) * len(PER_STOCK)
    print(f"共 {len(ids)} 檔 × {len(PER_STOCK)} 種資料，已完成 {total - len(todo)}／{total}")

    done = 0
    for ds, sid in todo:
        if time.time() > deadline:
            print("時間到，下次接著抓")
            break
        try:
            df = fetch(ds, sid)
        except Exception as e:
            print(f"  {ds} {sid} 失敗：{e}（下次重抓）")
            continue
        os.makedirs(f"{DATA}/{ds}", exist_ok=True)
        df.to_csv(f"{DATA}/{ds}/{sid}.csv.gz", index=False, compression="gzip")
        done += 1
        if done % 100 == 0:
            print(f"  本次已抓 {done} 個，剩 {len(todo) - done}")

    remaining = len(todo) - done
    if limit:  # 測試用的小量下載不更新進度，避免被誤判為已抓完
        print(f"測試模式：本次抓了 {done} 個檔案")
        return
    json.dump({"stocks": len(ids), "files_total": total, "files_done": total - remaining,
               "remaining": remaining, "updated": date.today().isoformat()},
              open(f"{DATA}/_progress.json", "w"), indent=2)
    print(f"本次抓了 {done} 個檔案，還剩 {remaining} 個")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--minutes", type=int, default=330)
    a = ap.parse_args()
    main(a.limit, a.minutes)
