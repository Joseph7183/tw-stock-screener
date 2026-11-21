"""比較台灣有名 ETF 的含息報酬。用法：python etf_compare.py → results/etf_compare.csv

- 含息：除息日把現金股利加回
- 分割：價格突然變成約 1/k（k≥2）視為分割並校正；海外 ETF 沒有漲跌幅限制，其他大波動照實計算
- 每檔都跟 0050 在「同一段期間」比較，避免成立時間不同造成誤會
"""
import numpy as np
import pandas as pd

from download_etfs import ETFS


def total_return(sid):
    p = pd.read_csv(f"data/etf/TaiwanStockPrice_{sid}.csv.gz")
    p = p[p["close"] > 0].drop_duplicates("date")
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date")["close"].astype(float)
    cash = pd.Series(0.0, index=p.index)
    try:
        d = pd.read_csv(f"data/etf/TaiwanStockDividend_{sid}.csv.gz")
    except Exception:
        d = pd.DataFrame()
    for _, r in d.iterrows():
        amt = (r.get("CashEarningsDistribution") or 0) + (r.get("CashStatutorySurplus") or 0)
        ex = r.get("CashExDividendTradingDate")
        if amt and amt == amt and isinstance(ex, str):
            pos = p.index.searchsorted(pd.Timestamp(ex))
            if pos < len(p):
                cash.iloc[pos] += amt
    raw = p / (p.shift(1) - cash)
    r = raw - 1
    for k in range(2, 11):
        split = (raw * k - 1).abs() < 0.03
        r[split] = (raw * k - 1)[split]
        rsplit = (raw / k - 1).abs() < 0.03
        r[rsplit] = (raw / k - 1)[rsplit]
    return (1 + r.iloc[1:].fillna(0)).cumprod()


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0.9 else nav.iloc[-1] - 1, (nav / nav.cummax() - 1).min(), yrs


def main():
    navs = {sid: total_return(sid) for sid in ETFS}
    base = navs["0050"]
    end = min(n.index[-1] for n in navs.values())
    rows = []
    for sid, nav in navs.items():
        start = nav.index[0]
        a = nav[nav.index <= end]
        b = base[(base.index >= start) & (base.index <= end)]
        a, b = a / a.iloc[0], b / b.iloc[0]
        ra, dda, yrs = stats(a)
        rb, ddb, _ = stats(b)
        row = {"代號": sid, "名稱": ETFS[sid], "資料起日": start.date(), "年數": round(yrs, 1),
               "年化（或累積）": ra, "同期 0050": rb, "差距": ra - rb, "最大回撤": dda, "同期 0050 回撤": ddb}
        for y in (3, 5):
            s = end - pd.DateOffset(years=y)
            if start <= s:
                x = nav[(nav.index >= s) & (nav.index <= end)]
                z = base[(base.index >= s) & (base.index <= end)]
                row[f"近{y}年年化"] = (x.iloc[-1] / x.iloc[0]) ** (1 / y) - 1
                row[f"近{y}年 0050"] = (z.iloc[-1] / z.iloc[0]) ** (1 / y) - 1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/etf_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pct = [c for c in df.columns if c not in ("代號", "名稱", "資料起日", "年數")]
    print(df.assign(**{c: df[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}") for c in pct}).to_string(index=False))
    print("資料截止：", end.date())


if __name__ == "__main__":
    main()
