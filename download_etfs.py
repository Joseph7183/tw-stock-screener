"""下載台灣有名 ETF 的日股價與配息，存到 data/etf/。用法：python download_etfs.py"""
import os

from market_data import finmind

ETFS = {
    "0050": "元大台灣50", "006208": "富邦台50", "0056": "元大高股息", "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息", "00713": "元大台灣高息低波", "00929": "復華台灣科技優息",
    "00692": "富邦公司治理", "00850": "元大臺灣ESG永續", "00881": "國泰台灣5G+",
    "00891": "中信關鍵半導體", "00662": "富邦NASDAQ", "00646": "元大S&P500",
    "00757": "統一FANG+", "009820": "元大納斯達克精選",
}

if __name__ == "__main__":
    os.makedirs("data/etf", exist_ok=True)
    for sid, name in ETFS.items():
        for ds in ("TaiwanStockPrice", "TaiwanStockDividend"):
            df = finmind(ds, sid, years=20)
            df.to_csv(f"data/etf/{ds}_{sid}.csv.gz", index=False)
            print(f"{sid} {name} {ds}: {len(df)} 筆")
