import json
with open('dataset/raw_generated.jsonl', 'r', encoding='utf-8') as f:
    line = f.readline()
    d = json.loads(line)
    for m in d.get('messages', []):
        if m.get('role') == 'system':
            s = m.get('content', '')
            print('System content length:', len(s))
            print('Full content:')
            print(s)
            print()
            print('Has all 15 cats:', all(cat in s for cat in ['飲食','日常','交通','娛樂','醫療','教育','還款','薪水','獎金','零用錢','兼職','投資','利息','欠款回收','其他']))
