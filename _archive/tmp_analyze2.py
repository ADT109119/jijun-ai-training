import json, re, collections, sys, io
from filter_dataset import is_description_aligned

src = "dataset/raw_generated.jsonl"
out_path = "dataset/analysis_report.txt"

samples = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

cat_stats = collections.Counter()
cat_removed = collections.Counter()
removed_examples = collections.defaultdict(list)

for s in samples:
    msgs = s.get("messages", [])
    ass = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    tc = re.search(r"<tool_call>(.*?)</tool_call>", ass)
    if tc:
        obj = json.loads(tc.group(1))
        args = obj.get("args", obj)
        cat = args.get("category", "")
        desc = args.get("description", "")
        typ = args.get("type", "")
        cat_stats[cat] += 1
        if not is_description_aligned(cat, desc):
            cat_removed[cat] += 1
            if len(removed_examples[cat]) < 3:
                removed_examples[cat].append((typ, desc, args.get("amount", "")))

lines = []
lines.append(f"{'Category':<12} {'Total':>6} {'Removed':>8} {'Rate':>7}")
for cat in sorted(cat_stats, key=lambda c: -cat_removed[c]):
    t = cat_stats[cat]
    r = cat_removed[cat]
    rate = r / t * 100 if t else 0
    lines.append(f"{cat:<12} {t:>6} {r:>8} {rate:>6.1f}%")

lines.append("")
lines.append("=== Removed examples per category (top removed, only 15+ samples) ===")
for cat in sorted(cat_removed, key=lambda c: -cat_removed[c]):
    if cat_stats[cat] < 15:
        continue
    lines.append(f"")
    lines.append(f"[{cat}] removed {cat_removed[cat]}/{cat_stats[cat]}")
    for typ, desc, amt in removed_examples[cat]:
        lines.append(f"  ({typ}) {desc[:60]} | amount={amt}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Written to {out_path}")
