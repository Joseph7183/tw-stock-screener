"""第四輪回測：篩選器名單＋市值加權＋緩衝（見 results/backtest/preregistration_round4.md）。

需要先跑過 backtest.py（picks.csv）與 round3.py（r3_summary.csv，Q0 自建 0050）。
用法：python round4.py → results/backtest/r4_summary.csv、r4_yearly.csv、r4_nav.csv、r4_holdings.csv
"""
from datetime import date

import numpy as np
import pandas as pd

import backtest as bt
from hypotheses import load

OUT = bt.OUT
ENTRY_N, MAX_N, KEEP_RANK = 30, 40, 40


def capped(w, cap):
    w = w.copy()
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        w[~over] += extra * w[~over] / w[~over].sum()
    return w


def main():
    uni, stocks, R, px, tv, per = load()
    R = R.drop(columns="_CASH", errors="ignore")
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
    picks = pd.read_csv(f"{OUT}/picks.csv", dtype={"sid": str})
    picks.columns = ["d", "sid"] + list(picks.columns[2:])
    picks["d"] = pd.to_datetime(picks["d"])

    def run(exclude=(), cap=None):
        targets, held, prev_pass, log = {}, [], set(), []
        for d in rebal:
            i = days.get_loc(d) - 1
            cut = bt.cutoff_for(d)
            close = PX.iloc[:i + 1].ffill().iloc[-1]
            alive = set(PX.columns[(i - last_i.iloc[i]) <= 7])

            def mcap(s):
                c = stocks[s].get("capital")
                c = c[c.index <= cut] if c is not None else []
                return close[s] * c.iloc[-1] / 10 if len(c) else np.nan

            passers = [s for s in picks.loc[picks["d"] == d, "sid"] if s not in exclude and s in alive]
            pm = pd.Series({s: mcap(s) for s in passers}).dropna().sort_values(ascending=False)
            rank = pd.Series(range(1, len(pm) + 1), index=pm.index)
            keep = [s for s in held if s in alive and
                    ((s in rank.index and rank[s] <= KEEP_RANK) or (s not in rank.index and s in prev_pass))]
            fill = [s for s in pm.index if s not in keep][:max(0, ENTRY_N - len(keep))]
            hold = pd.Series({s: mcap(s) for s in keep + fill}).dropna().sort_values(ascending=False).iloc[:MAX_N]
            w = hold / hold.sum() if len(hold) else pd.Series({"_CASH": 1.0})
            if cap and len(hold):
                w = capped(w, cap)
            targets[d] = w
            log.append({"d": d.date(), "持股數": len(hold), "緩衝中": sum(s not in rank.index for s in hold.index),
                        "最大持股": hold.index[0] if len(hold) else "", "最大權重": w.max(),
                        "台積電權重": w.get("2330", 0.0), "名單": "、".join(hold.index)})
            held, prev_pass = list(hold.index), set(pm.index)
        return targets, pd.DataFrame(log)

    R2 = R.copy()
    R2["_CASH"] = 0.0
    variants = {"P1 篩選器＋市值加權＋緩衝": run(),
                "P2 診斷：同 P1，單檔上限 20%": run(cap=0.20),
                "P3 診斷：同 P1，排除台積電": run(exclude=("2330",))}
    navs, rows = {}, {}
    for name, (t, log) in variants.items():
        nav, turn = bt.simulate(R2, t)
        navs[name], rows[name] = nav, bt.stats(nav, turn)
        rows[name]["平均持股數"] = log["持股數"].mean()
        rows[name]["平均台積電權重"] = log["台積電權重"].mean()
        rows[name]["平均最大權重"] = log["最大權重"].mean()
    variants["P1 篩選器＋市值加權＋緩衝"][1].to_csv(f"{OUT}/r4_holdings.csv", index=False, encoding="utf-8-sig")

    r50 = bt.daily_returns(bt.read("TaiwanStockPrice", "0050"), bt.read("TaiwanStockDividend", "0050"))
    nav50 = (1 + r50[r50.index >= rebal[0]]).cumprod() * (1 - bt.FEE)
    navs["真實 0050（含息）"], rows["真實 0050（含息）"] = nav50, bt.stats(nav50)
    q0 = pd.read_csv(f"{OUT}/r3_nav.csv", index_col=0, parse_dates=True)["Q0 自建 0050"]
    navs["Q0 自建 0050"], rows["Q0 自建 0050"] = q0, bt.stats(q0)

    nav_df = pd.DataFrame(navs).ffill().dropna()
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
    p1 = "P1 篩選器＋市值加權＋緩衝"
    ok = all(S.loc[p1, h] > S.loc[["真實 0050（含息）", "Q0 自建 0050"], h].max() for h in (H1, H2))

    S.to_csv(f"{OUT}/r4_summary.csv", encoding="utf-8-sig")
    yearly.T.to_csv(f"{OUT}/r4_yearly.csv", encoding="utf-8-sig")
    nav_df.to_csv(f"{OUT}/r4_nav.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(S.map(lambda x: f"{x:.1%}" if isinstance(x, float) and abs(x) < 20 else x).to_string())
    print(yearly.T.map(lambda x: f"{x:.1%}").to_string())
    print(variants[p1][1][["d", "持股數", "緩衝中", "最大持股", "最大權重", "台積電權重"]].to_string())
    print(f"\nP1 兩段都贏真實 0050 與自建 0050：{'是' if ok else '否'}")


if __name__ == "__main__":
    main()
