"""最精簡的 Notion API 包裝：查詢持股總覽、更新頁面、新增頁面。"""
import os
import time

import requests

from config import HOLDINGS_DS, NOTION_VERSION

API = "https://api.notion.com/v1"


def _headers():
    return {
        "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _request(method, path, **kw):
    for attempt in range(5):
        r = requests.request(method, f"{API}{path}", headers=_headers(), timeout=30, **kw)
        if r.status_code == 429:  # Notion 限速，等一下再試
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Notion {method} {path} 失敗：{r.status_code} {r.text[:300]}")
        return r.json()
    raise RuntimeError("Notion 連續限速，請稍後再跑")


def query_holdings():
    """回傳 {股票代號: {"page_id", "status", "asset_type"}}。"""
    out, cursor = {}, None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        data = _request("POST", f"/data_sources/{HOLDINGS_DS}/query", json=body)
        for page in data["results"]:
            p = page["properties"]
            title = "".join(t["plain_text"] for t in p["股票代號"]["title"]).strip()
            status = (p.get("狀態", {}).get("status") or {}).get("name")
            asset = (p.get("資產類型", {}).get("select") or {}).get("name")
            if title:
                out[title] = {"page_id": page["id"], "status": status, "asset_type": asset}
        if not data.get("has_more"):
            return out
        cursor = data["next_cursor"]


def update_page(page_id, props):
    _request("PATCH", f"/pages/{page_id}", json={"properties": props})


def create_page(props):
    _request("POST", "/pages", json={
        "parent": {"type": "data_source_id", "data_source_id": HOLDINGS_DS},
        "properties": props,
    })


# ── 屬性格式小工具 ───────────────────────────────────────
def num(v):
    return {"number": None if v is None else round(float(v), 4)}


def text(s):
    return {"rich_text": [{"text": {"content": str(s)[:1900]}}]}


def title(s):
    return {"title": [{"text": {"content": str(s)}}]}


def select(name):
    return {"select": None if name is None else {"name": name}}


def status(name):
    return {"status": {"name": name}}


def date(iso):
    return {"date": {"start": iso}}
