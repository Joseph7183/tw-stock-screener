"""第一次設定好後先跑這個，確認資料抓得到、科目對得上。

    python inspect_stock.py 2330
"""
import sys

from fundamentals import ITEMS, compute_metrics, pick
from market_data import finmind

sid = sys.argv[1] if len(sys.argv) > 1 else "2330"
income = finmind("TaiwanStockFinancialStatements", sid)
balance = finmind("TaiwanStockBalanceSheet", sid)
cashflow = finmind("TaiwanStockCashFlowsStatement", sid)

for label, df, keys in [("損益表", income, ["revenue", "gross", "op_income", "net_income"]),
                        ("資產負債表", balance, ["equity", "assets", "liabilities", "capital"]),
                        ("現金流量表", cashflow, ["ocf", "capex"])]:
    print(f"\n== {label}：{len(df)} 筆 ==")
    for k in keys:
        s = pick(df, k)
        print(f"  {k:12s} {'✓' if not s.empty else '✗ 找不到'}  最新：{s.iloc[-1] if not s.empty else '-'}")

m = compute_metrics(income, balance, cashflow)
print("\n== 計算結果 ==")
for k, v in m.items():
    print(f"  {k}: {v}")
print("\n有 ✗ 的話，把上面的輸出貼給 Claude，對照 FinMind 的 type 名稱修正 fundamentals.py 的 ITEMS。")
