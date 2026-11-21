#!/bin/bash
# 美股資料中途存檔（GitHub Actions 用）
git config user.name "github-actions"
git config user.email "github-actions@users.noreply.github.com"
git add data/us || exit 0
git diff --cached --quiet && exit 0
git commit -q -m "美股資料 $(date +%F-%H%M)"
for i in 1 2 3; do git pull -q --rebase origin main && git push -q && break; sleep 5; done
