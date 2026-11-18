"""每個交易日收盤後執行：更新持股總覽裡 Active／Watch 的現價、今日漲跌、價格更新日。
只用證交所／櫃買中心的免費資料，不耗 FinMind 額度。"""
import math
from datetime import datetime, timedelta, timezone

import notion_api as nt
from market_data import market_snapshot


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct_change(close, change):
    if not (ok(close) and ok(change)) or close == change:
        return None
    return change / (close - change)


def main():
    holdings = nt.query_holdings()
    targets = {k: v for k, v in holdings.items() if v["status"] in ("Active", "Watch")}
    if not targets:
        print("沒有 Active／Watch 的持股，結束。")
        return
    today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    snap = market_snapshot().set_index("stock_id")
    for sid, v in targets.items():
        if sid not in snap.index or not ok(snap.at[sid, "close"]):
            print(f"{sid} 找不到今日收盤價（非台股、停牌或今天休市），略過")
            continue
        close = snap.at[sid, "close"]
        chg = pct_change(close, snap.at[sid, "change"]) if "change" in snap else None
        nt.update_page(v["page_id"], {
            "現價（NT$）": nt.num(float(close)),
            "今日漲跌": nt.num(chg),
            "價格更新日": nt.date(today),
        })
        print(f"{sid} → {close}（{'' if chg is None else f'{chg:+.2%}'}）")


if __name__ == "__main__":
    main()
