"""回測：把選股系統當成一個「指數編製規則」，看能不能贏 0050。

做法（全部只用當時已經公開的資料，避免偷看未來）：
- 換股日：每年 4/1（年報）、5/16（Q1）、8/15（Q2）、11/15（Q3），財報公告期限隔天
- 每個換股日：用當時能看到的財報＋前一交易日股價／本益比，跑跟每月選股一模一樣的關卡和評分
- 已下市公司也在名單裡（避免倖存者偏差），下市後以最後價格轉成現金
- 報酬含現金股利、股票股利；股票分割／減資自動校正
- 成本：元大手續費 0.1425% × 6 折，賣出另收證交稅 0.3%（ETF 0.1%）

用法：python backtest.py            → 結果存到 results/backtest/
"""
import glob
import os
from datetime import date

import numpy as np
import pandas as pd

from config import GATES, SCORE_WEIGHTS
from fundamentals import ITEMS, check_gates, _annual_sum, _year_end, _ttm

DATA = "data"
OUT = "results/backtest"
START_YEAR = 2018                 # 資產負債表 2012 起，但 2012 歸屬母公司淨利缺季，5 年 ROE 最早 2018/4 可算
FEE = 0.001425 * 0.6
TAX_STOCK, TAX_ETF = 0.003, 0.001
TOP_N = 30
BUFFER_N = 45                     # 緩衝區：已持有的只要還通過且排名在前 45 就續抱


# ───────────────────────── 讀資料 ─────────────────────────
def read(ds, sid):
    f = f"{DATA}/{ds}/{sid}.csv.gz"
    if not os.path.exists(f) or os.path.getsize(f) < 40:
        return pd.DataFrame()
    try:
        return pd.read_csv(f, dtype={"stock_id": str})
    except Exception:
        return pd.DataFrame()


def pick_series(df):
    """一次把所有科目取出來（跟 fundamentals.pick 同邏輯），之後只要依日期切。"""
    out = {}
    if df.empty:
        return out
    df = df[~df["type"].astype(str).str.endswith("_per")]
    for item, (types, names) in ITEMS.items():
        s = pd.Series(dtype=float)
        for t in types:
            hit = df[df["type"] == t]
            if not hit.empty:
                s = hit.groupby("date")["value"].first().astype(float).sort_index()
                break
        if s.empty and "origin_name" in df:
            for n in names:
                hit = df[df["origin_name"].astype(str).str.contains(n, regex=False)]
                if not hit.empty:
                    s = hit.groupby("date")["value"].first().astype(float).sort_index()
                    break
        if not s.empty:
            s.index = pd.to_datetime(s.index)
            out[item] = s
    return out


def daily_returns(price, div):
    """含息日報酬。除權息日的參考價 = (昨收 − 現金股利) ÷ (1 + 股票股利/10)。
    漲跌幅限制 10%，超過代表分割／減資：能對上整數倍就校正，對不上就當 0。"""
    p = price[price["close"] > 0][["date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.drop_duplicates("date").set_index("date")["close"].astype(float)
    if len(p) < 2:
        return pd.Series(dtype=float)
    cash = pd.Series(0.0, index=p.index)
    stock = pd.Series(0.0, index=p.index)
    if not div.empty:
        for _, r in div.iterrows():
            c = (r.get("CashEarningsDistribution") or 0) + (r.get("CashStatutorySurplus") or 0)
            s = (r.get("StockEarningsDistribution") or 0) + (r.get("StockStatutorySurplus") or 0)
            for amt, col, ser in ((c, "CashExDividendTradingDate", cash), (s, "StockExDividendTradingDate", stock)):
                d = r.get(col)
                if amt and amt == amt and isinstance(d, str):
                    d = pd.Timestamp(d)
                    pos = p.index.searchsorted(d)
                    if pos < len(p):
                        ser.iloc[pos] += amt
    prev = p.shift(1)
    ref = (prev - cash) / (1 + stock / 10)
    raw = p / ref
    r = raw - 1
    big = (r.abs() > 0.105)
    k_down = np.round(1 / raw)            # 分割：價格變成 1/k
    k_up = np.round(raw)                  # 減資：價格變成 k 倍
    fix_down = big & (k_down >= 2) & ((raw * k_down - 1).abs() < 0.105)
    fix_up = big & (k_up >= 2) & ((raw / k_up - 1).abs() < 0.105)
    r[fix_down] = (raw * k_down - 1)[fix_down]
    r[fix_up] = (raw / k_up - 1)[fix_up]
    r[big & ~fix_down & ~fix_up] = 0.0
    return r.iloc[1:].fillna(0.0)


def load_all():
    uni = pd.read_csv(f"{DATA}/universe.csv", dtype={"stock_id": str})
    uni = uni[~uni["industry_category"].astype(str).str.contains("金融")]
    norm = uni["industry_category"].astype(str).str.replace(r"[類業]$", "", regex=True)
    uni["ind"] = norm.where(uni["industry_category"] != "已下市", None)
    stocks, rets, px, tv, per = {}, {}, {}, {}, {}
    for i, sid in enumerate(uni["stock_id"], 1):
        price = read("TaiwanStockPrice", sid)
        if price.empty:
            continue
        r = daily_returns(price, read("TaiwanStockDividend", sid))
        if r.empty:
            continue
        price["date"] = pd.to_datetime(price["date"])
        price = price.drop_duplicates("date").set_index("date")
        rets[sid] = r
        px[sid] = price["close"].astype(float)
        tv[sid] = price["Trading_money"].astype(float)
        pe = read("TaiwanStockPER", sid)
        if not pe.empty:
            pe["date"] = pd.to_datetime(pe["date"])
            per[sid] = pe.drop_duplicates("date").set_index("date")["PER"].astype(float)
        inc = pick_series(read("TaiwanStockFinancialStatements", sid))
        bal = pick_series(read("TaiwanStockBalanceSheet", sid))
        cf = pick_series(read("TaiwanStockCashFlowsStatement", sid))
        mr = read("TaiwanStockMonthRevenue", sid)
        if not mr.empty:
            mr["avail"] = pd.to_datetime(mr["date"]) + pd.Timedelta(days=9)   # 次月 10 日前公布
            mr = mr.sort_values(["revenue_year", "revenue_month"])
        stocks[sid] = {**inc, **{k: v for k, v in bal.items() if k in ("equity", "assets", "liabilities", "capital")},
                       **{k: v for k, v in cf.items() if k in ("ocf", "capex")}, "rev_m": mr}
        if i % 300 == 0:
            print(f"  讀取 {i}/{len(uni)}")
    R = pd.DataFrame(rets).sort_index().fillna(0.0)
    return uni.set_index("stock_id"), stocks, R, px, tv, per


# ───────────────────────── 時點財報指標 ─────────────────────────
def cutoff_for(d):
    """換股日 → 當時能看到的最新一季財報季底。"""
    y = d.year
    if d.month < 5 or (d.month == 5 and d.day < 16):
        return pd.Timestamp(y - 1, 12, 31)
    if d.month < 8 or (d.month == 8 and d.day < 15):
        return pd.Timestamp(y, 3, 31)
    if d.month < 11 or (d.month == 11 and d.day < 15):
        return pd.Timestamp(y, 6, 30)
    return pd.Timestamp(y, 9, 30)


def metrics_at(s, cut):
    """跟 fundamentals.compute_metrics 同一套算法，只是先把財報切到 cut 為止。"""
    g = {k: v[v.index <= cut] for k, v in s.items() if k != "rev_m"}
    e = pd.Series(dtype=float)
    rev, gp, ni = g.get("revenue", e), g.get("gross", e), g.get("net_income", e)
    eq, assets, liab, cap = g.get("equity", e), g.get("assets", e), g.get("liabilities", e), g.get("capital", e)
    ocf, capex = g.get("ocf", e), g.get("capex", e)
    ni_y, rev_y, gp_y = _annual_sum(ni), _annual_sum(rev), _annual_sum(gp)
    eq_y, ocf_y, capex_y = _year_end(eq), _year_end(ocf), _year_end(capex)
    years = sorted(set(ni_y.index) & set(eq_y.index))[-GATES["years"]:]
    roe_y = {}
    for y in years:
        avg_eq = (eq_y[y] + eq_y[y - 1]) / 2 if (y - 1) in eq_y.index else eq_y[y]
        roe_y[y] = ni_y[y] / avg_eq if avg_eq > 0 else np.nan
    gm_y = (gp_y / rev_y).dropna()
    gm_y = gm_y[gm_y.index.isin(years)]
    fcf_y = {y: ocf_y[y] - abs(capex_y.get(y, 0.0)) for y in years if y in ocf_y.index}
    ocf_sum = sum(ocf_y[y] for y in years if y in ocf_y.index)
    ni_sum = sum(ni_y[y] for y in years)
    return {
        "years": years,
        "roe_min": min(roe_y.values()) if roe_y else np.nan,
        "roe_avg": float(np.nanmean(list(roe_y.values()))) if roe_y else np.nan,
        "gm_std": float(gm_y.std()) if len(gm_y) >= 3 else np.nan,
        "loss_years": sum(1 for y in years if ni_y[y] <= 0),
        "fcf_positive_years": sum(1 for v in fcf_y.values() if v > 0),
        "ocf_to_ni": ocf_sum / ni_sum if ni_sum > 0 else np.nan,
        "debt_ratio": liab.iloc[-1] / assets.iloc[-1] if len(liab) and len(assets) else np.nan,
        "shares": cap.iloc[-1] / 10 if len(cap) else np.nan,
    }


def rev_growth_3m(mr, d):
    if mr is None or mr.empty:
        return np.nan
    s = mr[mr["avail"] <= d]["revenue"].astype(float).tolist()
    if len(s) < 15:
        return np.nan
    prev = sum(s[-15:-12])
    return sum(s[-3:]) / prev - 1 if prev > 0 else np.nan


def last_before(series, d):
    if series is None:
        return np.nan
    s = series[series.index < d]
    return s.iloc[-1] if len(s) else np.nan


def screen_at(d, uni, stocks, px, tv, per):
    """在換股日 d 跑一次完整選股，回傳通過者（含評分）。"""
    cut = cutoff_for(d)
    rows = []
    for sid in stocks:
        p = px[sid]
        before = p[p.index < d]
        if before.empty or (d - before.index[-1]).days > 10:      # 已下市或停牌
            continue
        rows.append({"sid": sid, "close": before.iloc[-1], "pe": last_before(per.get(sid), d),
                     "ind": uni.at[sid, "ind"] if sid in uni.index else None})
    df = pd.DataFrame(rows)
    v = df[df["pe"] > 0]
    ind_pct = v.groupby("ind")["pe"].rank(pct=True)
    mkt_pct = v["pe"].rank(pct=True)
    peers = v.groupby("ind")["pe"].transform("count")
    df["pe_pct"] = ind_pct.where((peers >= 5) & v["ind"].notna(), mkt_pct)
    # 跟每月選股一樣先用便宜條件初篩
    cand = df[(df["pe"] > 0) & (df["pe"] < GATES["max_pe"]) & (df["pe_pct"] <= GATES["max_pe_industry_pct"])]
    out = []
    for r in cand.itertuples():
        m = metrics_at(stocks[r.sid], cut)
        if len(m["years"]) < GATES["years"] or m["loss_years"] > 0:
            continue
        t = tv[r.sid]
        avg_tv = t[t.index < d].tail(60).mean()
        ok, _, mcap = check_gates(m, r.close, r.pe, r.pe_pct, avg_tv)
        if ok:
            out.append({"sid": r.sid, "pe_pct": r.pe_pct, "roe_avg": m["roe_avg"], "gm_std": m["gm_std"],
                        "rev_growth_3m": rev_growth_3m(stocks[r.sid]["rev_m"], d), "mcap": mcap})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    pct = lambda s, asc=True: s.rank(pct=True, ascending=asc).fillna(0.5)
    w = SCORE_WEIGHTS
    res["q"] = (pct(res["roe_avg"]) + pct(res["gm_std"], False)) / 2 * w["quality"]
    res["v"] = (1 - res["pe_pct"].fillna(0.5)) * w["value"]
    res["m"] = pct(res["rev_growth_3m"]) * w["momentum"]
    res["score"] = res["q"] + res["v"] + res["m"]
    return res.sort_values("score", ascending=False).set_index("sid")


# ───────────────────────── 組合規則 ─────────────────────────
def eq_w(ids):
    return pd.Series(1 / len(ids), index=ids) if len(ids) else pd.Series(dtype=float)


def score_w(sc):
    return sc / sc.sum() if len(sc) else pd.Series(dtype=float)


def rule_all(p, prev):            return eq_w(p.index)
def rule_top30(p, prev):          return eq_w(p.index[:TOP_N])
def rule_top30_score(p, prev):    return score_w(p["score"].iloc[:TOP_N])
def rule_score50(p, prev):        return score_w(p.loc[p["score"] >= 50, "score"])


def rule_top30_buffer(p, prev):
    """已持有的若仍通過且排名在前 45 就續抱，空位再由高分補滿 30 檔。"""
    rank = pd.Series(range(1, len(p) + 1), index=p.index)
    keep = [s for s in prev if s in rank.index and rank[s] <= BUFFER_N][:TOP_N]
    fill = [s for s in p.index if s not in keep][:TOP_N - len(keep)]
    return eq_w(keep + fill)


def rule_top30_score_buffer(p, prev):
    rank = pd.Series(range(1, len(p) + 1), index=p.index)
    keep = [s for s in prev if s in rank.index and rank[s] <= BUFFER_N][:TOP_N]
    fill = [s for s in p.index if s not in keep][:TOP_N - len(keep)]
    return score_w(p.loc[keep + fill, "score"])


def rule_top30_mcap(p, prev):
    """前 30 名依市值加權（跟 0050 同一種加權法），單檔上限 20%。"""
    w = score_w(p["mcap"].iloc[:TOP_N])
    for _ in range(10):
        over = w > 0.2
        if not over.any():
            break
        extra = (w[over] - 0.2).sum()
        w[over] = 0.2
        w[~over] += extra * w[~over] / w[~over].sum()
    return w


STRATEGIES = {
    "A 全部通過者・等權": rule_all,
    "B 前30名・等權": rule_top30,
    "C 前30名・評分加權": rule_top30_score,
    "D 評分≥50・評分加權": rule_score50,
    "E 前30名・等權＋緩衝": rule_top30_buffer,
    "F 前30名・評分加權＋緩衝": rule_top30_score_buffer,
    "G 前30名・市值加權（上限20%）": rule_top30_mcap,
}


def simulate(R, targets, cost=True, tax=TAX_STOCK, freq=1):
    """逐日模擬。targets: {交易日: 權重(可含 _CASH)}，freq=2 代表每兩次才換一次。
    換股日以收盤價成交，隔天開始承擔報酬。"""
    rebal = set(sorted(targets)[::freq])
    days = R.index[R.index >= min(targets)]
    hold, navs, turn = pd.Series({"_CASH": 1.0}), {}, []
    for d in days:
        hold = hold * (1 + R.loc[d, hold.index])
        nav = hold.sum()
        if d in rebal:
            tgt = targets[d] * nav
            idx = tgt.index.union(hold.index).drop("_CASH", errors="ignore")
            diff = tgt.reindex(idx, fill_value=0) - hold.reindex(idx, fill_value=0)
            buys, sells = diff[diff > 0].sum(), -diff[diff < 0].sum()
            c = (buys * FEE + sells * (FEE + tax)) if cost else 0.0
            turn.append((buys + sells) / 2 / nav)
            hold = targets[d] * (nav - c)
            nav -= c
        navs[d] = nav
    return pd.Series(navs), turn


def stats(nav, turn=None):
    nav = nav / nav.iloc[0]
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    dr = nav.pct_change().dropna()
    dd = (nav / nav.cummax() - 1).min()
    return {
        "年化報酬": nav.iloc[-1] ** (1 / yrs) - 1,
        "累積報酬": nav.iloc[-1] - 1,
        "年化波動": dr.std() * np.sqrt(252),
        "最大回撤": dd,
        "夏普(無風險0)": dr.mean() / dr.std() * np.sqrt(252) if dr.std() else np.nan,
        "每次平均換手": np.mean(turn) if turn else np.nan,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print("讀取資料…")
    uni, stocks, R, px, tv, per = load_all()
    R["_CASH"] = 0.0
    days = R.index
    print(f"共 {len(stocks)} 檔，交易日 {days[0].date()} ~ {days[-1].date()}")

    # 換股日：財報期限隔天後的第一個交易日
    rebal_dates = []
    for y in range(START_YEAR, date.today().year + 1):
        for m, dd in ((4, 1), (5, 16), (8, 15), (11, 15)):
            t = pd.Timestamp(y, m, dd)
            pos = days.searchsorted(t)
            if pos < len(days) and days[pos] < days[-1]:
                rebal_dates.append(days[pos])

    print(f"換股 {len(rebal_dates)} 次，開始逐次選股…")
    picks = {}
    for d in rebal_dates:
        p = screen_at(d, uni, stocks, px, tv, per)
        picks[d] = p
        print(f"  {d.date()} 通過 {len(p)} 檔" + (f"，最高分 {p.index[0]} {p['score'].iloc[0]:.1f}" if len(p) else ""))
    pd.concat({d.date(): p for d, p in picks.items() if len(p)}).to_csv(f"{OUT}/picks.csv", encoding="utf-8-sig")

    def cashify(w):
        return w if len(w) else pd.Series({"_CASH": 1.0})

    results, navs = {}, {}
    for name, rule in STRATEGIES.items():
        targets, prev = {}, []
        for d in rebal_dates:
            w = rule(picks[d], prev) if len(picks[d]) else pd.Series(dtype=float)
            prev = list(w.index)
            targets[d] = cashify(w)
        nav, turn = simulate(R, targets)
        navs[name] = nav
        results[name] = stats(nav, turn)
        nav0, _ = simulate(R, targets, cost=False)
        results[name]["未扣成本年化"] = stats(nav0)["年化報酬"]
        if name.startswith("B"):
            nav_h, turn_h = simulate(R, targets, freq=2)
            navs["B' 前30名・等權・半年換一次"] = nav_h
            results["B' 前30名・等權・半年換一次"] = stats(nav_h, turn_h)
            nav_y, turn_y = simulate(R, targets, freq=4)
            navs["B'' 前30名・等權・一年換一次"] = nav_y
            results["B'' 前30名・等權・一年換一次"] = stats(nav_y, turn_y)

    # 基準：0050（含息、買進持有）、全市場等權（每季再平衡，不扣成本）
    first = rebal_dates[0]
    r50 = daily_returns(read("TaiwanStockPrice", "0050"), read("TaiwanStockDividend", "0050"))
    r50 = r50[r50.index >= first]
    nav50 = (1 + r50).cumprod() * (1 - FEE)
    navs["基準 0050（含息）"] = nav50
    results["基準 0050（含息）"] = stats(nav50)
    targets = {}
    for d in rebal_dates:
        alive = [s for s in stocks if s in R.columns and px[s][px[s].index < d].size
                 and (d - px[s][px[s].index < d].index[-1]).days <= 10]
        targets[d] = eq_w(alive)
    nav_ew, _ = simulate(R, targets, cost=False)
    navs["基準 全市場等權"] = nav_ew
    results["基準 全市場等權"] = stats(nav_ew)

    # 評分三等分：分數高的真的比較好嗎？
    for k, label in enumerate(["高分組", "中分組", "低分組"]):
        targets = {}
        for d in rebal_dates:
            p = picks[d]
            n = len(p)
            ids = p.index[k * n // 3:(k + 1) * n // 3] if n >= 3 else []
            targets[d] = cashify(eq_w(list(ids)))
        nav_t, _ = simulate(R, targets, cost=False)
        navs[f"評分{label}（通過者三等分）"] = nav_t
        results[f"評分{label}（通過者三等分）"] = stats(nav_t)

    nav_df = pd.DataFrame(navs).ffill()
    nav_df = nav_df / nav_df.iloc[0]
    nav_df.to_csv(f"{OUT}/nav.csv", encoding="utf-8-sig")
    summary = pd.DataFrame(results).T
    # 前後半段：檢查是不是只在某段時間有效
    mid = pd.Timestamp(2022, 1, 1)
    for label, sl in (("2018/4–2021", nav_df[nav_df.index < mid]), ("2022–至今", nav_df[nav_df.index >= mid])):
        yrs = (sl.index[-1] - sl.index[0]).days / 365.25
        summary[f"年化 {label}"] = (sl.iloc[-1] / sl.iloc[0]) ** (1 / yrs) - 1
    yearly = nav_df.resample("YE").last()
    yearly = pd.concat([nav_df.iloc[[0]], yearly]).pct_change().iloc[1:]
    yearly.index = yearly.index.year
    summary.to_csv(f"{OUT}/summary.csv", encoding="utf-8-sig")
    yearly.T.to_csv(f"{OUT}/yearly.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summary.map(lambda x: f"{x:.1%}" if isinstance(x, float) and abs(x) < 50 else x).to_string())
    print(yearly.T.map(lambda x: f"{x:.1%}").to_string())


if __name__ == "__main__":
    main()
