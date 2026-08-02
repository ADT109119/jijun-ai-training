import io
import re
import os
import csv

LOGS = [
    ("json_fixed", r"C:\Users\me\AppData\Local\Temp\opencode\benchmark_json.log",
     "JSON + 50 label fixes (2026-08-02)", "saves/best_bookkeeping_model.pt"),
    ("compressed_fixed", r"C:\Users\me\AppData\Local\Temp\opencode\benchmark_compressed.log",
     "Compressed + 50 label fixes (2026-08-02)", "saves/compressed/best_bookkeeping_model.pt"),
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")

TIER = re.compile(r"難度分層: (\S+)")
TOTAL = re.compile(r"測試樣本數: (\d+) 筆")
FMT = re.compile(r"格式通過率: ([\d.]+)% \((\d+)/(\d+)\)")
EM = re.compile(r"完全匹配率: ([\d.]+)% \((\d+)/(\d+)\)")
AMT = re.compile(r"金額正確率: ([\d.]+)%")
CAT = re.compile(r"分類正確率: ([\d.]+)%")
ACC = re.compile(r"帳戶正確率: ([\d.]+)%")
DATE = re.compile(r"日期正確率: ([\d.]+)%")
REWARD = re.compile(r"平均 RL Reward 分數: ([\d.]+)")
LAT = re.compile(r"平均推論延遲: ([\d.]+)")
GT = re.compile(r"總計統計 \((\d+) 筆")
GFMT = re.compile(r"總格式通過率 \(Format Pass Rate\): ([\d.]+)%")
GEM = re.compile(r"總精準匹配率 \(Exact Match Rate\)\s*: ([\d.]+)%")

rows = []
for commit, path, desc, model in LOGS:
    with io.open(path, encoding="utf-16") as f:
        txt = f.read()

    tier_totals = []
    tier = {"name": None}
    grand_total = grand_fmt = grand_em = None
    for line in txt.splitlines():
        s = line.strip()
        m = TIER.search(s)
        if m:
            tier = {"name": m.group(1)}
            continue
        m = TOTAL.search(s)
        if m:
            tier["total"] = int(m.group(1))
            tier_totals.append(tier)
            continue
        for rx, key in ((FMT, "fmt"), (EM, "em")):
            m = rx.search(s)
            if m:
                tier[key] = float(m.group(1))
                break
        for rx, key in ((AMT, "amt"), (CAT, "cat"), (ACC, "acc"), (DATE, "date")):
            m = rx.search(s)
            if m:
                tier[key] = float(m.group(1))
                break
        m = REWARD.search(s)
        if m:
            tier["reward"] = float(m.group(1))
        m = LAT.search(s)
        if m:
            tier["latency"] = float(m.group(1))
        m = GFMT.search(s)
        if m:
            grand_fmt = float(m.group(1))
        m = GEM.search(s)
        if m:
            grand_em = float(m.group(1))

    grand_total = sum(t["total"] for t in tier_totals)
    # reconstruct raw counts from printed pct (2dp) to aggregate per-field
    def agg(key):
        cnt = 0
        for t in tier_totals:
            cnt += round(t[key] / 100.0 * t["total"])
        return cnt / grand_total * 100.0

    rows.append({
        "commit": commit,
        "model": model,
        "format_pass_rate": grand_fmt,
        "exact_match_rate": grand_em,
        "amount_acc": round(agg("amt"), 2),
        "category_acc": round(agg("cat"), 2),
        "account_acc": round(agg("acc"), 2),
        "date_acc": round(agg("date"), 2),
        "avg_reward": round(sum(t["reward"] * t["total"] for t in tier_totals) / grand_total, 2),
        "avg_latency_ms": round(sum(t["latency"] * t["total"] for t in tier_totals) / grand_total, 1),
        "status": "keep",
        "description": desc,
    })
    print(commit, rows[-1])

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", OUT)
