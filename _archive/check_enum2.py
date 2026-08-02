import json, re

with open('dataset/raw_generated.jsonl', 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]

print('Samples:', len(lines))
for i, line in enumerate(lines[:3]):
    d = json.loads(line)
    print('\n=== Sample %d ===' % (i+1))
    for m in d.get('messages', []):
        role = m.get('role', '')
        if role == 'system':
            # Find category enum
            m2 = re.search(r'"category"[^}]+"enum":\s*(\[[^\]]+\])', m.get('content', ''))
            if m2:
                cats = json.loads(m2.group(1))
                print('System enum (%d cats):' % len(cats), cats)
            else:
                print('System content preview:', m.get('content', '')[:200])
        elif role == 'assistant':
            c = m.get('content', '')[:100]
            print('Assistant:', c)
