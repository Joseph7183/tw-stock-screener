"""把 FinMind 的季報整理成年度指標，並判斷是否通過嚴格關卡。

注意 FinMind 的資料特性：
- 損益表：單季數字 → 全年 = 四季相加
- 資產負債表：時點數字 → 年底 = 12/31 那筆
- 現金流量表：年初累計到該季 → 全年 = 12/31 那筆
"""
import numpy as np
import pandas as pd

from config import GATES

# 每個科目：先找 type，找不到再用中文科目名（origin_name）的關鍵字
ITEMS = {
    "revenue":   (["Revenue", "OperatingRevenue"], ["營業收入合計", "營業收入"]),
    "gross":     (["GrossProfit"], ["營業毛利"]),
    "op_income": (["OperatingIncome"], ["營業利益"]),
    "net_income": (["EquityAttributableToOwnersOfParent", "IncomeAfterTaxes"],
                   ["歸屬於母公司業主", "本期淨利"]),
    "equity":    (["EquityAttributableToOwnersOfParent", "Equity"],
                  ["歸屬於母公司業主之權益", "權益總"]),
    "assets":    (["TotalAssets"], ["資產總"]),
    "liabilities": (["Liabilities", "TotalLiabilities"], ["負債總"]),
    "capital":   (["OrdinaryShare", "CapitalStock", "CommonStock"], ["普通股股本", "股本"]),
    "ocf":       (["CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities"],
                  ["營業活動之淨現金"]),
    "capex":     (["PropertyAndPlantAndEquipment", "AcquisitionOfPropertyPlantAndEquipment"],
                  ["取得不動產、廠房及設備"]),
}


def pick(df, item):
    """從長表格取出某科目，回傳以日期為索引的 Series。"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    types, names = ITEMS[item]
    df = df[~df["type"].astype(str).str.endswith("_per")]
    for t in types:
        hit = df[df["type"] == t]
        if not hit.empty:
            return hit.groupby("date")["value"].first().astype(float).sort_index()
    if "origin_name" in df:
        for n in names:
            hit = df[df["origin_name"].astype(str).str.contains(n, regex=False)]
            if not hit.empty:
                return hit.groupby("date")["value"].first().astype(float).sort_index()
    return pd.Series(dtype=float)


def _annual_sum(s):
    """單季數字 → 每年合計（只保留四季都有的年度）。"""
    if s.empty:
        return s
    years = pd.to_datetime(s.index).year
    g = s.groupby(years)
    return g.sum()[g.count() == 4]


def _year_end(s):
    """取每年 12/31 的數字。"""
    if s.empty:
        return s
    idx = pd.to_datetime(s.index)
    mask = (idx.month == 12) & (idx.day == 31)
    out = s[mask]
    out.index = pd.to_datetime(out.index).year
    return out


def _ttm(s):
    return s.iloc[-4:].sum() if len(s) >= 4 else np.nan


def compute_metrics(income, balance, cashflow):
    """回傳年度與近四季指標 dict。"""
    n = GATES["years"]
    rev, gp, oi, ni = (pick(income, k) for k in ("revenue", "gross", "op_income", "net_income"))
    eq, assets, liab, cap = (pick(balance, k) for k in ("equity", "assets", "liabilities", "capital"))
    ocf, capex = pick(cashflow, "ocf"), pick(cashflow, "capex")

    ni_y, rev_y, gp_y = _annual_sum(ni), _annual_sum(rev), _annual_sum(gp)
    eq_y, ocf_y, capex_y = _year_end(eq), _year_end(ocf), _year_end(capex)

    years = sorted(set(ni_y.index) & set(eq_y.index))[-n:]
    roe_y = {}
    for y in years:
        avg_eq = (eq_y[y] + eq_y[y - 1]) / 2 if (y - 1) in eq_y.index else eq_y[y]
        roe_y[y] = ni_y[y] / avg_eq if avg_eq > 0 else np.nan

    gm_y = (gp_y / rev_y).dropna()
    gm_y = gm_y[gm_y.index.isin(years)]
    fcf_y = {y: ocf_y[y] - abs(capex_y.get(y, 0.0)) for y in years if y in ocf_y.index}
    ocf_sum = sum(ocf_y[y] for y in years if y in ocf_y.index)
    ni_sum = sum(ni_y[y] for y in years)

    latest_eq = eq.iloc[-1] if not eq.empty else np.nan
    return {
        "years": years,
        "roe_by_year": roe_y,
        "roe_min": min(roe_y.values()) if roe_y else np.nan,
        "roe_avg": float(np.nanmean(list(roe_y.values()))) if roe_y else np.nan,
        "roe_ttm": _ttm(ni) / latest_eq if latest_eq and latest_eq > 0 else np.nan,
        "gross_margin_ttm": _ttm(gp) / _ttm(rev) if _ttm(rev) else np.nan,
        "op_margin_ttm": _ttm(oi) / _ttm(rev) if _ttm(rev) else np.nan,
        "gm_std": float(gm_y.std()) if len(gm_y) >= 3 else np.nan,
        "loss_years": sum(1 for y in years if ni_y[y] <= 0),
        "fcf_positive_years": sum(1 for v in fcf_y.values() if v > 0),
        "ocf_to_ni": ocf_sum / ni_sum if ni_sum > 0 else np.nan,
        "debt_ratio": liab.iloc[-1] / assets.iloc[-1] if not liab.empty and not assets.empty else np.nan,
        "shares": cap.iloc[-1] / 10 if not cap.empty else np.nan,  # 面額 10 元
    }


def check_gates(m, close, pe, pe_pct, avg_trade_value):
    """回傳 (是否通過, 未通過原因清單)。缺資料也算未通過。"""
    g, fails = GATES, []

    def need(ok, msg):
        if not ok:
            fails.append(msg)

    def bad(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    need(len(m["years"]) >= g["years"], f"財報不足 {g['years']} 年")
    mcap = close * m["shares"] if not bad(m["shares"]) and close else np.nan
    need(not bad(mcap) and mcap >= g["min_market_cap"], "市值不足 100 億")
    need(not bad(avg_trade_value) and avg_trade_value >= g["min_avg_trade_value"], "成交值不足")
    need(m["loss_years"] == 0, "有虧損年度")
    need(not bad(m["roe_min"]) and m["roe_min"] >= g["min_roe_each_year"], "有年度 ROE < 10%")
    need(not bad(m["roe_avg"]) and m["roe_avg"] >= g["min_roe_avg"], "平均 ROE < 15%")
    need(m["fcf_positive_years"] >= g["min_fcf_positive_years"], "自由現金流為正年數不足")
    need(not bad(m["ocf_to_ni"]) and m["ocf_to_ni"] >= g["min_ocf_to_ni"], "現金流品質不足（營業現金流／淨利）")
    need(not bad(m["debt_ratio"]) and m["debt_ratio"] < g["max_debt_ratio"], "負債比 ≥ 50%")
    need(not bad(m["gm_std"]) and m["gm_std"] <= g["max_gross_margin_std"], "毛利率波動太大")
    need(not bad(pe) and 0 < pe < g["max_pe"], "本益比 ≥ 30 或為負")
    need(not bad(pe_pct) and pe_pct <= g["max_pe_industry_pct"], "本益比高於同業 75 百分位")
    return (not fails), fails, mcap


def revenue_growth(month_rev):
    """回傳 (近 12 個月營收年增率, 近 3 個月營收年增率)。"""
    if month_rev is None or month_rev.empty:
        return np.nan, np.nan
    s = month_rev.sort_values(["revenue_year", "revenue_month"])["revenue"].astype(float).tolist()

    def yoy(k):
        if len(s) < 12 + k:
            return np.nan
        prev = sum(s[-12 - k:-12])
        return sum(s[-k:]) / prev - 1 if prev > 0 else np.nan

    return yoy(12) if len(s) >= 24 else np.nan, yoy(3)
