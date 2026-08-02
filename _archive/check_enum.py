import json, re

with open('dataset/raw_generated.jsonl', 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]

print('Samples:', len(lines))
if lines:
    d = json.loads(lines[0])
    for m in d.get('messages', []):
        if m.get('role') == 'system':
            s = m.get('content', '')
            m2 = re.search(r'"enum":\s*(\[[^\]]+\])', s)
            if m2:
                cats = json.loads(m2.group(1))
                print('Enum has', len(cats), 'categories')
                for c in cats:
                    print('  -', c)
