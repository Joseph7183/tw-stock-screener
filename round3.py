"""第三輪回測：0050 去掉爛公司（見 results/backtest/preregistration_round3.md）。

用法：python round3.py → results/backtest/r3_summary.csv、r3_yearly.csv、r3_nav.csv
"""
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

import backtest as bt

OUT = bt.OUT
CACHE = "/tmp/r3_cache.pkl"


def load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    uni = pd.read_csv(f"{bt.DATA}/universe.csv", dtype={"stock_id": str})
    uni = uni[uni["type"].isna() | (uni["type"] == "twse")]          # 上市；已下市者類別不明一併納入
    rets, px, fin = {}, {}, {}
    for i, sid in enumerate(uni["stock_id"], 1):
        price = bt.read("TaiwanStockPrice", sid)
        if price.empty:
            continue
        r = bt.daily_returns(price, bt.read("TaiwanStockDividend", sid))
        if r.empty:
            continue
        price["date"] = pd.to_datetime(price["date"])
        rets[sid] = r
        px[sid] = price.drop_duplicates("date").set_index("date")["close"].astype(float)
        inc = bt.pick_series(bt.read("TaiwanStockFinancialStatements", sid))
        bal = bt.pick_series(bt.read("TaiwanStockBalanceSheet", sid))
        fin[sid] = {k: v for k, v in (("net_income", inc.get("net_income")), ("equity", bal.get("equity")),
                                       ("capital", bal.get("capital"))) if v is not None}
        if i % 300 == 0:
            print(f"  讀取 {i}/{len(uni)}")
    data = (pd.DataFrame(rets).sort_index().fillna(0.0), px, fin)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def is_bad(f, cut):
    """近 5 個完整年度有虧損，或平均 ROE < 10%。資料不足（0 年）→ 不算爛。"""
    m = bt.metrics_at(f, cut)
    if not m["years"]:
        return False
    return m["loss_years"] > 0 or (m["roe_avg"] == m["roe_avg"] and m["roe_avg"] < 0.10)


def main():
    R, px, fin = load()
    days = R.index
    PX = pd.DataFrame(px).reindex(days)
    n = np.arange(len(days))
    last_i = PX.notna().mul(n, axis=0).where(PX.notna()).ffill()
    rebal = []
    for y in range(bt.START_YEAR, date.today().year + 1):
        for m, dd in ((4, 1), (5, 16), (8, 15), (11, 15)):
            pos = days.searchsorted(pd.Timestamp(y, m, dd))
            if pos < len(days) and days[pos] < days[-1]:
                rebal.append(days[pos])

    T = {"Q0": {}, "Q1": {}, "Q2": {}}
    log = []
    for d in rebal:
        i = days.get_loc(d) - 1
        cut = bt.cutoff_for(d)
        alive = PX.columns[(i - last_i.iloc[i]) <= 7]
        close = PX.iloc[:i + 1].ffill().iloc[-1]
        mcap = {}
        for sid in alive:
            cap = fin[sid].get("capital")
            if cap is None:
                continue
            cap = cap[cap.index <= cut]
            if len(cap):
                mcap[sid] = close[sid] * cap.iloc[-1] / 10
        mcap = pd.Series(mcap).sort_values(ascending=False)
        top50 = mcap.iloc[:50]
        bad = {s for s in mcap.index[:120] if is_bad(fin[s], cut)}
        q1 = top50[[s not in bad for s in top50.index]]
        q2 = mcap[[s not in bad for s in mcap.index]].iloc[:50]
        T["Q0"][d], T["Q1"][d], T["Q2"][d] = (s / s.sum() for s in (top50, q1, q2))
        log.append({"d": d.date(), "剔除": "、".join(s for s in top50.index if s in bad),
                    "剔除權重": 1 - q1.sum() / top50.sum(), "台積電權重Q0": T["Q0"][d].get("2330", 0)})
        print(f"  {d.date()} 前50大剔除 {len(top50) - len(q1)} 檔（權重 {log[-1]['剔除權重']:.1%}），台積電 {log[-1]['台積電權重Q0']:.0%}")

    R2 = R.copy()
    R2["_CASH"] = 0.0
    names = {"Q0": "Q0 自建 0050", "Q1": "Q1 去掉爛公司（不補）", "Q2": "Q2 去掉爛公司（補滿）"}
    navs, rows = {}, {}
    for k, t in T.items():
        nav, turn = bt.simulate(R2, t)
        navs[names[k]], rows[names[k]] = nav, bt.stats(nav, turn)
    r50 = bt.daily_returns(bt.read("TaiwanStockPrice", "0050"), bt.read("TaiwanStockDividend", "0050"))
    nav50 = (1 + r50[r50.index >= rebal[0]]).cumprod() * (1 - bt.FEE)
    navs["真實 0050（含息）"], rows["真實 0050（含息）"] = nav50, bt.stats(nav50)

    nav_df = pd.DataFrame(navs).ffill()
    nav_df = nav_df / nav_df.iloc[0]
    S = pd.DataFrame(rows).T
    mid = pd.Timestamp(2022, 1, 1)
    H1, H2 = "前半 2018/4–2021", "後半 2022–至今"
    for label, sl in ((H1, nav_df[nav_df.index < mid]), (H2, nav_df[nav_df.index >= mid])):
        yrs = (sl.index[-1] - sl.index[0]).days / 365.25
        S[label] = (sl.iloc[-1] / sl.iloc[0]) ** (1 / yrs) - 1
    yearly = nav_df.resample("YE").last()
    yearly = pd.concat([nav_df.iloc[[0]], yearly]).pct_change().iloc[1:]
    yearly.index = yearly.index.year

    cand = S.loc[[names["Q1"], names["Q2"]], H1].idxmax()
    bars = S.loc[["真實 0050（含息）", names["Q0"]]]
    ok = all(S.loc[cand, h] > bars[h].max() for h in (H1, H2))

    S.to_csv(f"{OUT}/r3_summary.csv", encoding="utf-8-sig")
    yearly.T.to_csv(f"{OUT}/r3_yearly.csv", encoding="utf-8-sig")
    nav_df.to_csv(f"{OUT}/r3_nav.csv", encoding="utf-8-sig")
    pd.DataFrame(log).to_csv(f"{OUT}/r3_excluded.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(S.map(lambda x: f"{x:.1%}" if isinstance(x, float) else x).to_string())
    print(yearly.T.map(lambda x: f"{x:.1%}").to_string())
    print(f"\n候選：{cand}；兩段都贏真實 0050 與自建 0050：{'是' if ok else '否'}")


if __name__ == "__main__":
    main()
