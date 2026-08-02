import json, re
import generate_dataset as G
import filter_dataset as F

TOOL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)

EXPENSE_BASE = set(G.EXPENSE_CATS)
INCOME_BASE = set(G.INCOME_CATS)
ALL_BASE = set(G.EXPENSE_CATS) | set(G.INCOME_CATS)

EXP_SIG = ['餐','吃','喝','買','付','繳','花','訂','購','房租','管理費','水電','學費','課程','電影','車票','掛號','零食','外套','球鞋','書','保養','加油','門票','便當','咖啡','晚餐','早餐','午餐','報名','網購','刷卡','帳單','飲料','便當','電暖','除濕','修繕','漏水']
INC_SIG = ['薪','獎金','收入','還款','還錢','償','利息','股利','兼職','打工','收到','入帳','匯入','生活費','零用錢','尾款','賣出','賣','獲利','紅包','壓歲錢','稿費','演出費','退款','配息','酬勞']

def label_args(d):
    ass = d['messages'][-1].get('content', '')
    mt = TOOL_RE.search(ass)
    if not mt: return None
    return json.loads(mt.group(1)).get('args', {})

def cat_type(cat):
    if cat in EXPENSE_BASE: return 'expense'
    if cat in INCOME_BASE: return 'income'
    return None

def best_match(desc, cats):
    best, best_n = None, 0
    for c in cats:
        if c == '其他': continue
        kws = F.CATEGORY_KEYWORDS.get(c, [])
        n = sum(1 for kw in kws if len(kw) >= 2 and kw.lower() in desc.lower())
        if n > best_n:
            best, best_n = c, n
    return best, best_n

MANUAL = {
    '購買兩本經典文學': ('教育', 'expense'),
    '買書': ('教育', 'expense'),
    '購買新電暖器': ('日常', 'expense'),
    '購買除濕機': ('日常', 'expense'),
    '購買老布鞋': ('日常', 'expense'),
    '網購外套': ('日常', 'expense'),
    '網購運動鞋': ('日常', 'expense'),
    '購買外套': ('日常', 'expense'),
    '買外套': ('日常', 'expense'),
    '晚餐帶孫子吃飯': ('飲食', 'expense'),
    '下班買零食': ('飲食', 'expense'),
    '早餐店三明治豆漿': ('飲食', 'expense'),
    '修繕浴室漏水': ('日常', 'expense'),
    '午餐': ('飲食', 'expense'),
    '購買參考書與拿鐵': ('教育', 'expense'),
    '線上理財課程學費': ('教育', 'expense'),
    '報名線上課程': ('教育', 'expense'),
    '線上課程報名費': ('教育', 'expense'),
    '買線上課程': ('教育', 'expense'),
    '晚餐外送': ('飲食', 'expense'),
    '商業午餐': ('飲食', 'expense'),
    '晚餐': ('飲食', 'expense'),
    '早餐店三明治飲料試吃反饋金': ('其他', 'expense'),
}

changes = []
lines_out = []
n = 0
for line in open('dataset/test_strata.jsonl', encoding='utf-8'):
    if not line.strip():
        lines_out.append(line); continue
    n += 1
    d = json.loads(line)
    a = label_args(d)
    if not a:
        lines_out.append(line); continue
    cat, typ, desc = a.get('category',''), a.get('type',''), a.get('description','')
    new_cat, new_type = cat, typ

    e_sig = sum(1 for s in EXP_SIG if s in desc)
    i_sig = sum(1 for s in INC_SIG if s in desc)

    # manual override first
    if desc in MANUAL:
        new_cat, new_type = MANUAL[desc]
    else:
        if e_sig >= 2 and i_sig == 0:
            new_type = 'expense'
            if cat not in EXPENSE_BASE and cat != '其他':
                b, bn = best_match(desc, EXPENSE_BASE)
                new_cat = b if b else '其他'
            elif cat == '其他':
                pass
        elif i_sig >= 2 and e_sig == 0:
            new_type = 'income'
            if cat not in INCOME_BASE and cat != '其他':
                b, bn = best_match(desc, INCOME_BASE)
                new_cat = b if b else '其他'
            elif cat == '其他':
                pass
        else:
            ok, reason = F.is_description_aligned(cat, desc)
            if not ok and reason.startswith('other_kw:'):
                tgt = reason.split(':',1)[1]
                if tgt in ALL_BASE:
                    new_cat = tgt
                    new_type = cat_type(tgt) or typ

    if new_cat != cat or new_type != typ:
        a['category'] = new_cat
        a['type'] = new_type
        # rewrite assistant tool_call
        new_args = json.dumps(a, ensure_ascii=False)
        old_ass = d['messages'][-1]['content']
        new_ass = re.sub(r'<tool_call>.*?</tool_call>', f'<tool_call>{{"name": "add_record", "args": {new_args}}}</tool_call>', old_ass, flags=re.S)
        d['messages'][-1]['content'] = new_ass
        changes.append((n, cat, typ, desc, '->', new_cat, new_type))
    lines_out.append(json.dumps(d, ensure_ascii=False) + '\n')

with open('dataset/test_strata.jsonl', 'w', encoding='utf-8') as f:
    f.writelines(lines_out)

print(f'共 {n} 筆，修正 {len(changes)} 筆：')
for c in changes:
    print(f'  L{c[0]}: [{c[1]}/{c[2]}] {c[3]}  ->  [{c[5]}/{c[6]}]')
