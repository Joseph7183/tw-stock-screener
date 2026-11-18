"""每個交易日收盤後執行：把持股總覽裡 Active／Watch 的現價更新成今天收盤價。
只用證交所／櫃買中心的免費資料，不耗 FinMind 額度。"""
import notion_api as nt
from market_data import market_snapshot


def main():
    holdings = nt.query_holdings()
    targets = {k: v for k, v in holdings.items() if v["status"] in ("Active", "Watch")}
    if not targets:
        print("沒有 Active／Watch 的持股，結束。")
        return
    snap = market_snapshot().set_index("stock_id")
    for sid, v in targets.items():
        if sid in snap.index and snap.at[sid, "close"]:
            nt.update_page(v["page_id"], {"現價（NT$）": nt.num(snap.at[sid, "close"])})
            print(f"{sid} → {snap.at[sid, 'close']}")
        else:
            print(f"{sid} 找不到今日收盤價（非台股、停牌或今天休市），略過")


if __name__ == "__main__":
    main()
