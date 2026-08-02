import json, re, collections

# Analyze kept data distribution
kept = []
with open("dataset/raw_filtered.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            kept.append(json.loads(line))

kept_cats = collections.Counter()
kept_types = collections.Counter()
kept_extra = collections.Counter()

base_cats = {"飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款", "薪水", "獎金", "零用錢", "兼職", "投資", "利息", "欠款回收", "其他"}

for s in kept:
    msgs = s.get("messages", [])
    ass = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    tc = re.search(r"<tool_call>(.*?)</tool_call>", ass)
    if tc:
        obj = json.loads(tc.group(1))
        args = obj.get("args", obj)
        cat = args.get("category", "")
        typ = args.get("type", "")
        kept_cats[cat] += 1
        kept_types[typ] += 1
        if cat not in base_cats:
            kept_extra[cat] += 1

lines = []
lines.append(f"Total kept: {len(kept)}")
lines.append(f"Type distribution: {dict(kept_types)}")
lines.append("")
lines.append("Base category distribution (kept):")
for cat in sorted(base_cats):
    lines.append(f"  {cat}: {kept_cats.get(cat, 0)}")

lines.append("")
lines.append(f"Extra (pool) categories used: {len(kept_extra)} total")
for cat in sorted(kept_extra):
    lines.append(f"  {cat}: {kept_extra[cat]}")

with open("dataset/kept_distribution.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Written to dataset/kept_distribution.txt")
