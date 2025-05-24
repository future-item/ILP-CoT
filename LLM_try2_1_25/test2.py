#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download all positives into …/positive/  and all negatives into …/negative/
• 自动重试、断点续传
• 如下载失败或校验不一致会删除损坏文件
• 可反复运行 —— 已完整文件跳过，残缺文件续传
"""

import sys, time, importlib
from pathlib import Path
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter, Retry

# ======================== 配置 ========================= #
ROOT = Path("/home/pengyf/Third_Work/Work/ILP-CoT-Customization")
CONNECT_TIMEOUT = 10          # TCP/SSL 建链超时
READ_TIMEOUT    = 300         # 单次读取超时
MAX_RETRY       = 5           # 自动重试次数
CHUNK           = 1 << 14     # 16 KiB
REFRESH_EVERY   = 128         # 进度刷新块数
# ====================================================== #

# ---------- helper: 字节转可读 ----------
def h(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

# ---------- 打印工具 ----------
def ok(p):
    print(f"\r\033[32m✔ {p} [{h(p.stat().st_size)}]\033[0m".ljust(70))
def fail(name,e):
    print(f"\r\033[31m✘ {name} -> {e}\033[0m".ljust(70), file=sys.stderr)
def progress(name, done, total, start):
    pct  = done/total*100 if total else 0
    rate = done/(time.time()-start+1e-3)
    bar  = f"{pct:6.1f}% {h(done):>7}/{h(total) if total else '?'} @ {h(rate)}/s"
    print(f"\r{name:<30}{bar}", end="", flush=True)

# ---------- 带重试 Session ----------
def new_session():
    s = requests.Session()
    retry = Retry(total=MAX_RETRY, backoff_factor=1,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s
S = new_session()

# ---------- 核心下载 ----------
def download(url: str, dst: Path) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    try:
        with S.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                   headers=headers, stream=True) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            total = r.headers.get("Content-Length")
            if total is not None:
                total = int(total) + (tmp.stat().st_size if headers else 0)

            mode  = "ab" if headers else "wb"
            start = time.time()
            with open(tmp, mode) as f:
                for i, chunk in enumerate(r.iter_content(CHUNK), 1):
                    if chunk:
                        f.write(chunk)
                        if i % REFRESH_EVERY == 0:
                            progress(dst.name, f.tell(), total, start)

        if total and tmp.stat().st_size != total:
            raise RuntimeError("size mismatch")
        tmp.rename(dst)
        ok(dst)
        return True

    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        fail(dst.name, e)
        return False

# ---------- 主流程 ----------
def main():
    Prepration = importlib.import_module("Prepration")
    POS_DIR = ROOT / "positive"
    NEG_DIR = ROOT / "negative"
    failed = []

    for idx in range(22):
        device, url_pos_root, url_neg_root, \
        pos_names, neg_names, *_ = Prepration.prepration(idx)

        print(f"\n== Batch {idx} | +{len(pos_names)}  -{len(neg_names)} ==")

        for name in pos_names:
            url = urljoin(url_pos_root.rstrip("/") + "/", name)
            dst = POS_DIR / name
            if dst.exists() and dst.stat().st_size:
                ok(dst); continue
            if not download(url, dst):
                failed.append(url)

        for name in neg_names:
            url = urljoin(url_neg_root.rstrip("/") + "/", name)
            dst = NEG_DIR / name
            if dst.exists() and dst.stat().st_size:
                ok(dst); continue
            if not download(url, dst):
                failed.append(url)

    # ------ 汇总 ------
    print("\n========== SUMMARY ==========")
    if failed:
        print(f"\033[31m未成功下载 {len(failed)} 张：\033[0m")
        for u in failed:
            print("  ", u)
        sys.exit(1)
    else:
        print("\033[32m全部图片下载完成！\033[0m")

if __name__ == "__main__":
    main()
