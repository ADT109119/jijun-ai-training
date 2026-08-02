import json, re, random
import filter_dataset as F

random.seed(0)
removed = []
for p in ['dataset/qwen_full/raw_generated.jsonl', 'dataset/qwen_supp/raw_generated.jsonl']:
    for line in open(p, encoding='utf-8'):
        if line.strip():
            s = json.loads(line)
            ass = next((m['content'] for m in s['messages'] if m['role'] == 'assistant'), '')
            tc = re.search(r'<tool_call>(.*?)</tool_call>', ass, re.S)
            if tc:
                o = json.loads(tc.group(1)); a = o.get('args', o)
                cat = a.get('category', ''); desc = a.get('description', '')
                if not F.is_description_aligned(cat, desc):
                    removed.append((cat, desc, a.get('amount'), a.get('type')))

print(f'Total removed: {len(removed)}')
print('=== random sample of 30 removed ===')
for cat, desc, amt, typ in random.sample(removed, min(30, len(removed))):
    print(f'  [{cat:8s}] {str(desc):28s} ${amt} {str(typ):7s}')
