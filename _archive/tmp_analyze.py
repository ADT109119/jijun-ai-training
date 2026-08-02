import json, re, collections, sys
from filter_dataset import is_description_aligned

src = "dataset/raw_generated.jsonl"
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
                removed_examples[cat].append((typ, desc, args.get("amount","")))

print(f"{'Category':<10} {'Total':>6} {'Removed':>8} {'Rate':>7}")
for cat in sorted(cat_stats, key=lambda c: -cat_removed[c]):
    t = cat_stats[cat]
    r = cat_removed[cat]
    rate = r / t * 100 if t else 0
    print(f"{cat:<10} {t:>6} {r:>8} {rate:>6.1f}%")

print("\n=== Removed examples per category (top removed) ===")
for cat in sorted(cat_removed, key=lambda c: -cat_removed[c])[:12]:
    print(f"\n[{cat}] removed {cat_removed[cat]}/{cat_stats[cat]}")
    for typ, desc, amt in removed_examples[cat]:
        print(f"  ({typ}) {desc[:50]} | amount={amt}")
