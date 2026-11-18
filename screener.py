"""每月執行一次：全市場嚴格篩選，結果寫回 Notion 持股總覽，並存一份 CSV 紀錄。

流程：
1. 證交所／櫃買中心快照 → 初篩（一般股、非金融、本益比、同業百分位、成交值）
2. 通過初篩的公司逐檔抓 FinMind 財報 → 嚴格關卡
3. 通過者計算選股評分 → 前 N 名新增為 Watch；已在帳本裡的個股一律更新指標
"""
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

import notion_api as nt
from config import EXCLUDED_INDUSTRIES, GATES, INDUSTRY_MAP, MAX_NEW_WATCH, SCORE_WEIGHTS
from fundamentals import check_gates, compute_metrics, revenue_growth
from market_data import finmind, market_snapshot, stock_info


def prefilter(snap, info, keep):
    """只用免費全市場資料做的便宜初篩。keep 裡的代號（帳本裡的個股）一律保留。"""
    df = snap[snap["stock_id"].str.fullmatch(r"[1-9]\d{3}")].copy()
    df["industry"] = df["stock_id"].map(info["industry_category"]) if not info.empty else None
    df = df[~df["industry"].isin(EXCLUDED_INDUSTRIES)]
    valid = df["pe"].notna() & (df["pe"] > 0)
    v = df[valid]
    industry_pct = v.groupby("industry")["pe"].rank(pct=True)
    market_pct = v["pe"].rank(pct=True)
    peers = v.groupby("industry")["pe"].transform("count")
    # 同業不到 5 家時百分位沒意義，改跟全市場比
    df["pe_pct"] = industry_pct.where(peers >= 5, market_pct)
    loose = (valid & (df["pe"] < GATES["max_pe"])
             & (df["pe_pct"] <= GATES["max_pe_industry_pct"])
             & (df["trade_value"] >= GATES["min_avg_trade_value"] * 0.5))
    return df[loose | df["stock_id"].isin(keep)]


def evaluate(row):
    """逐檔抓財報並跑關卡。提早淘汰以節省 FinMind 額度。"""
    sid = row.stock_id
    income = finmind("TaiwanStockFinancialStatements", sid)
    balance = finmind("TaiwanStockBalanceSheet", sid)
    m = compute_metrics(income, balance, None)
    roe_bad = m["roe_min"] == m["roe_min"] and m["roe_min"] < GATES["min_roe_each_year"]
    if len(m["years"]) < GATES["years"] or m["loss_years"] > 0 or roe_bad:
        ok, fails, mcap = check_gates(m, row.close, row.pe, row.pe_pct, np.nan)
        checked = ("財報不足", "有虧損年度", "有年度 ROE", "平均 ROE", "負債比", "毛利率", "本益比", "市值")
        fails = [f for f in fails if f.startswith(checked)]
        if len(m["years"]) < GATES["years"]:
            fails = [f"財報不足 {GATES['years']} 年或抓不到資料"]
        return m, False, fails, mcap, np.nan, np.nan, np.nan
    cashflow = finmind("TaiwanStockCashFlowsStatement", sid)
    m = compute_metrics(income, balance, cashflow)
    price = finmind("TaiwanStockPrice", sid, years=0.3)
    avg_tv = price["Trading_money"].astype(float).tail(60).mean() if "Trading_money" in price else np.nan
    ok, fails, mcap = check_gates(m, row.close, row.pe, row.pe_pct, avg_tv)
    g12 = g3 = np.nan
    if ok:
        g12, g3 = revenue_growth(finmind("TaiwanStockMonthRevenue", sid, years=2.5))
    return m, ok, fails, mcap, avg_tv, g12, g3


def score(df):
    """只在通過者之間排名：品質 40、價值 30、動能 30。"""
    def pct(s, ascending=True):
        return s.rank(pct=True, ascending=ascending).fillna(0.5)
    w = SCORE_WEIGHTS
    quality = (pct(df["roe_avg"]) + pct(df["gm_std"], ascending=False)) / 2
    value = 1 - df["pe_pct"].fillna(0.5)
    momentum = pct(df["rev_growth_3m"])
    return (quality * w["quality"] + value * w["value"] + momentum * w["momentum"]).round(1)


def to_props(r, today):
    def f(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else x
    return {
        "現價（NT$）": nt.num(f(r["close"])),
        "定量關卡": nt.select("通過" if r["passed"] else "未通過"),
        "未通過原因": nt.text("、".join(r["fails"])) if r["fails"] else {"rich_text": []},
        "選股評分": nt.num(f(r.get("score"))),
        "ROE": nt.num(f(r["roe_ttm"])),
        "毛利率": nt.num(f(r["gross_margin_ttm"])),
        "營益率": nt.num(f(r["op_margin_ttm"])),
        "負債比": nt.num(f(r["debt_ratio"])),
        "FCF為正年數": nt.num(r["fcf_positive_years"]),
        "本益比": nt.num(f(r["pe"])),
        "本益比同業百分位": nt.num(f(r["pe_pct"])),
        "殖利率": nt.num(f(r["div_yield"])),
        "營收成長（年）": nt.num(f(r["rev_growth_12m"])),
        "指標更新日": nt.date(today),
    }


def main(limit=None, dry_run=False):
    today = date.today().isoformat()
    holdings = nt.query_holdings()
    my_stocks = [k for k, v in holdings.items() if v["asset_type"] == "個股"]
    print(f"帳本裡的個股：{my_stocks or '無'}")

    snap, info = market_snapshot(), stock_info()
    cand = prefilter(snap, info, my_stocks)
    if limit:
        cand = pd.concat([cand[cand["stock_id"].isin(my_stocks)], cand.head(limit)]).drop_duplicates("stock_id")
    print(f"全市場 {len(snap)} 檔 → 初篩剩 {len(cand)} 檔，開始逐檔檢查財報…")

    results = []
    for i, row in enumerate(cand.itertuples(), 1):
        try:
            m, ok, fails, mcap, avg_tv, g12, g3 = evaluate(row)
        except Exception as e:  # 單檔失敗不影響整體
            print(f"  {row.stock_id} 抓取失敗：{e}")
            continue
        results.append({"stock_id": row.stock_id, "name": row.name, "market": row.market,
                        "industry": row.industry, "close": row.close, "pe": row.pe,
                        "pe_pct": row.pe_pct, "div_yield": row.div_yield, "market_cap": mcap,
                        "avg_trade_value": avg_tv, "passed": ok, "fails": fails,
                        "rev_growth_12m": g12, "rev_growth_3m": g3,
                        **{k: v for k, v in m.items() if k not in ("years", "roe_by_year")}})
        if i % 25 == 0:
            print(f"  進度 {i}/{len(cand)}，目前通過 {sum(r['passed'] for r in results)} 檔")

    df = pd.DataFrame(results)
    if df.empty:
        print("沒有任何結果，結束。")
        return
    df["score"] = np.nan
    passed = df[df["passed"]].copy()
    if not passed.empty:
        df.loc[passed.index, "score"] = score(passed)
    df = df.sort_values("score", ascending=False, na_position="last")

    os.makedirs("results", exist_ok=True)
    out = f"results/screen_{today}.csv"
    df.assign(fails=df["fails"].str.join("、")).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"通過 {len(passed)} 檔；完整結果存到 {out}")

    if dry_run:
        print(df[df["passed"]][["stock_id", "name", "score", "roe_avg", "pe"]].head(MAX_NEW_WATCH))
        return

    added = 0
    for r in df.to_dict("records"):
        props = to_props(r, today)
        if r["stock_id"] in holdings:
            nt.update_page(holdings[r["stock_id"]]["page_id"], props)
        elif r["passed"] and added < MAX_NEW_WATCH:
            props.update({
                "股票代號": nt.title(r["stock_id"]),
                "名稱": nt.text(r["name"] or ""),
                "狀態": nt.status("Watch"),
                "資產類型": nt.select("個股"),
                "投資定位": nt.select("觀察"),
                "交易所": nt.select(r["market"]),
                "產業": nt.select(INDUSTRY_MAP.get(r["industry"], r["industry"] or "其他")),
            })
            nt.create_page(props)
            added += 1
    print(f"Notion：更新 {sum(s in holdings for s in df['stock_id'])} 檔帳本個股，新增 {added} 檔 Watch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只檢查前 N 檔（測試用）")
    ap.add_argument("--dry-run", action="store_true", help="不寫入 Notion，只印結果")
    a = ap.parse_args()
    main(limit=a.limit, dry_run=a.dry_run)
