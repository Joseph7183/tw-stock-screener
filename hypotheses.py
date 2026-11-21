"""第二輪回測：四個事先登記的假說（見 results/backtest/preregistration.md）。

用法：python hypotheses.py   → results/backtest/h_summary.csv、h_yearly.csv、h_nav.csv
需要先跑過 backtest.py（H3 會用到它產生的 picks.csv）。
"""
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

import backtest as bt
from config import GATES

OUT = bt.OUT
CACHE = "/tmp/bt_cache.pkl"


def load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    data = bt.load_all()
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def rev_yoy(mr, d):
    """回傳 (近 3 個月年增率, 近 12 個月年增率)，只用 d 之前已公布的月營收。"""
    if mr is None or mr.empty:
        return np.nan, np.nan
    s = mr[mr["avail"] <= d]["revenue"].astype(float).tolist()

    def yoy(k):
        if len(s) < 12 + k:
            return np.nan
        prev = sum(s[-12 - k:-12])
        return sum(s[-k:]) / prev - 1 if prev > 0 else np.nan
    return yoy(3), yoy(12)


def main():
    uni, stocks, R, px, tv, per = load()
    R = R.drop(columns="_CASH", errors="ignore")
    days = R.index
    PX = pd.DataFrame(px).reindex(days)
    TV = pd.DataFrame(tv).reindex(days)
    TRI = (1 + R).cumprod()
    n = np.arange(len(days))
    last_i = PX.notna().mul(n, axis=0).where(PX.notna()).ffill()   # 最後一次有交易的位置

    rebal = []
    for y in range(bt.START_YEAR, date.today().year + 1):
        for m, dd in ((4, 1), (5, 16), (8, 15), (11, 15)):
            pos = days.searchsorted(pd.Timestamp(y, m, dd))
            if pos < len(days) and days[pos] < days[-1]:
                rebal.append(days[pos])

    picks = pd.read_csv(f"{OUT}/picks.csv", dtype={"sid": str})
    picks.columns = ["d", "sid"] + list(picks.columns[2:])
    picks["d"] = pd.to_datetime(picks["d"])

    targets = {h: {} for h in ("H1", "H2", "H3", "H4")}
    for d in rebal:
        i = days.get_loc(d) - 1                      # 只用前一交易日以前的資料
        cut = bt.cutoff_for(d)
        alive = (i - last_i.iloc[i]) <= 7            # 約 10 個日曆天
        hist = PX.iloc[max(0, i - 251):i + 1].notna().sum() >= 200
        close = PX.iloc[:i + 1].ffill().iloc[-1]
        avg_tv = TV.iloc[i - 59:i + 1].mean()
        liquid = []
        for sid in PX.columns[alive & hist & (avg_tv >= GATES["min_avg_trade_value"])]:
            cap = stocks[sid].get("capital")
            if cap is None:
                continue
            cap = cap[cap.index <= cut]
            if len(cap) and close[sid] * cap.iloc[-1] / 10 >= GATES["min_market_cap"]:
                liquid.append(sid)
        mom = (TRI.iloc[i - 21] / TRI.iloc[i - 252] - 1)[liquid]
        vol = R.iloc[i - 251:i + 1][liquid].std()

        def eq(ids):
            ids = list(ids)
            return pd.Series(1 / len(ids), index=ids) if ids else pd.Series({"_CASH": 1.0})

        targets["H1"][d] = eq(mom.nlargest(30).index)
        g = pd.DataFrame([rev_yoy(stocks[s]["rev_m"], d) for s in liquid], index=liquid, columns=["g3", "g12"])
        targets["H2"][d] = eq(g[g["g12"] > 0]["g3"].nlargest(30).index)
        passers = picks.loc[picks["d"] == d, "sid"].tolist()
        pm = (TRI.iloc[i - 21] / TRI.iloc[i - 252] - 1).reindex(passers).dropna()
        targets["H3"][d] = eq(pm.nlargest(15).index)
        targets["H4"][d] = eq(vol.nsmallest(30).index)
        print(f"  {d.date()} 流動股 {len(liquid)} 檔、通過關卡 {len(passers)} 檔")

    R2 = R.copy()
    R2["_CASH"] = 0.0
    names = {"H1": "H1 價格動能", "H2": "H2 營收動能", "H3": "H3 品質＋動能", "H4": "H4 低波動"}
    navs, rows = {}, {}
    for h, t in targets.items():
        nav, turn = bt.simulate(R2, t)
        navs[names[h]] = nav
        rows[names[h]] = bt.stats(nav, turn)
        rows[names[h]]["未扣成本年化"] = bt.stats(bt.simulate(R2, t, cost=False)[0])["年化報酬"]
        # 平均持有幾檔台積電權重（看是不是又靠台積電）
        rows[names[h]]["台積電入選比例"] = np.mean(["2330" in w.index for w in t.values()])

    r50 = bt.daily_returns(bt.read("TaiwanStockPrice", "0050"), bt.read("TaiwanStockDividend", "0050"))
    nav50 = (1 + r50[r50.index >= rebal[0]]).cumprod() * (1 - bt.FEE)
    navs["基準 0050（含息）"] = nav50
    rows["基準 0050（含息）"] = bt.stats(nav50)

    nav_df = pd.DataFrame(navs).ffill()
    nav_df = nav_df / nav_df.iloc[0]
    summary = pd.DataFrame(rows).T
    mid = pd.Timestamp(2022, 1, 1)
    for label, sl in (("前半 2018/4–2021", nav_df[nav_df.index < mid]), ("後半 2022–至今", nav_df[nav_df.index >= mid])):
        yrs = (sl.index[-1] - sl.index[0]).days / 365.25
        summary[label] = (sl.iloc[-1] / sl.iloc[0]) ** (1 / yrs) - 1
        dd = (sl / sl.cummax() - 1).min()
        summary[label.split()[0] + "回撤"] = dd
    yearly = nav_df.resample("YE").last()
    yearly = pd.concat([nav_df.iloc[[0]], yearly]).pct_change().iloc[1:]
    yearly.index = yearly.index.year

    # 判定（照事先登記的規則）
    hyp = summary.drop(index="基準 0050（含息）")
    b = summary.loc["基準 0050（含息）"]
    cand = hyp["前半 2018/4–2021"].idxmax()
    ok = (hyp.loc[cand, "前半 2018/4–2021"] > b["前半 2018/4–2021"]) and (hyp.loc[cand, "後半 2022–至今"] > b["後半 2022–至今"])

    nav_df.to_csv(f"{OUT}/h_nav.csv", encoding="utf-8-sig")
    summary.to_csv(f"{OUT}/h_summary.csv", encoding="utf-8-sig")
    yearly.T.to_csv(f"{OUT}/h_yearly.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summary.map(lambda x: f"{x:.1%}" if isinstance(x, float) else x).to_string())
    print(yearly.T.map(lambda x: f"{x:.1%}").to_string())
    print(f"\n候選：{cand}；兩段都贏 0050：{'是' if ok else '否'}")


if __name__ == "__main__":
    main()
