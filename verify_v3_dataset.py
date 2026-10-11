import json, re
from datetime import date

W = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]
rows = [json.loads(l) for l in open('dataset/v3/train_v3.jsonl', encoding='utf-8')]
augs = [r for l in [] for r in []] or [r for r in rows if r.get('aug')]

def sysparse(c):
    m = re.match(r'^今天是 (\d{4}-\d{2}-\d{2})（(星期.)）。', c)
    return m.groups()

def labdate(c):
    j = re.search(r'<tool_call>([\s\S]*?)</tool_call>', c)
    return json.loads(j.group(1))['args']['date']

# 抽樣：昨天 + week-3
sample = None
for r in augs:
    if '昨天' in r['messages'][1]['content'] and r['aug'] == 'week-3':
        sample = r
        break
print("aug:", sample['aug'])
print("user:", sample['messages'][1]['content'][:90])
ref, wd = sysparse(sample['messages'][0]['content'])
lab = labdate(sample['messages'][2]['content'])
print("ref:", ref, wd, "| label:", lab, "| offset days:",
      (date.fromisoformat(lab) - date.fromisoformat(ref)).days)
y, m, d = map(int, ref.split('-'))
print("星期覆核:", wd, "==", W[date(y, m, d).isoweekday() % 7],
      "->", wd == W[date(y, m, d).isoweekday() % 7])

# 全域：aug 樣本的 parent offset 一致性（day+2 => ref 位移+2, week-3 => -21）
# parent = aug 相同 messages 但無 aug 的樣本；以 (user, label-ref offset) 對照
by_user = {}
for r in rows:
    if not r.get('aug'):
        u = r['messages'][1]['content']
        refp, _ = sysparse(r['messages'][0]['content'])
        labp = labdate(r['messages'][2]['content'])
        by_user.setdefault(u, []).append((refp, labp))
viol = 0
checked = 0
for r in augs:
    u = r['messages'][1]['content']
    refc, wdc = sysparse(r['messages'][0]['content'])
    labc = labdate(r['messages'][2]['content'])
    off = (date.fromisoformat(labc) - date.fromisoformat(refc)).days
    delta = 2 if r['aug'] == 'day+2' else -21
    ok = False
    for (refp, labp) in by_user.get(u, []):
        offp = (date.fromisoformat(labp) - date.fromisoformat(refp)).days
        if off == offp and (date.fromisoformat(refc) - date.fromisoformat(refp)).days == delta:
            ok = True
            break
    checked += 1
    if not ok:
        viol += 1
print(f"aug parent-offset 恆等覆核: checked={checked} viol={viol}")

# 星期全線覆核
badwd = 0
for r in rows:
    ref2, wd2 = sysparse(r['messages'][0]['content'])
    y2, m2, d2 = map(int, ref2.split('-'))
    if W[date(y2, m2, d2).isoweekday() % 7] != wd2:
        badwd += 1
print("全線星期錯位:", badwd)
