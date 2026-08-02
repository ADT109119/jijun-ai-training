import json, re
from collections import Counter

cat_counts = Counter()
with open('dataset/raw_generated.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            for m in d.get('messages', []):
                if m.get('role') == 'system':
                    s = m.get('content', '')
                    m2 = re.search(r'"enum":\s*(\[[^\]]+\])', s)
                    if m2:
                        cats = json.loads(m2.group(1))
                        cat_counts[len(cats)] += 1

print('Enum size distribution:', dict(cat_counts))
