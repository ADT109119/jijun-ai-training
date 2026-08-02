import json, re
import generate_dataset as G
import filter_dataset as F

TOOL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
EXP_SIG = ['餐','吃','喝','買','付','繳','花','訂','購','房租','管理費','水電','學費','課程','電影','車票','掛號','零食','外套','球鞋','書','保養','加油','門票','便當','咖啡','晚餐','早餐','午餐','報名','網購','刷卡','帳單','飲料','電暖','除濕','修繕','漏水']
INC_SIG = ['薪','獎金','收入','還款','還錢','償','利息','股利','兼職','打工','收到','入帳','匯入','生活費','零用錢','尾款','賣出','獲利','紅包','壓歲錢','稿費','演出費','退款','配息','酬勞','賣']

def label_args(d):
    ass = d['messages'][-1].get('content', '')
    mt = TOOL_RE.search(ass)
    if not mt: return None
    return json.loads(mt.group(1)).get('args', {})

cands = []
for idx, line in enumerate(open('dataset/test_strata.jsonl', encoding='utf-8'), 1):
    if not line.strip(): continue
    d = json.loads(line)
    a = label_args(d)
    if not a: continue
    cat, typ, desc = a.get('category',''), a.get('type',''), a.get('description','')
    user_q = next((m.get('content','') for m in d['messages'] if m.get('role')=='user'), '')
    e_sig = sum(1 for s in EXP_SIG if s in desc)
    i_sig = sum(1 for s in INC_SIG if s in desc)
    ok, reason = F.is_description_aligned(cat, desc)
    strong = (e_sig >= 2 and i_sig == 0) or (i_sig >= 2 and e_sig == 0)
    sem = (not ok and reason.startswith('other_kw:'))
    if strong or sem:
        cands.append((idx, cat, typ, desc, reason if sem else f'strong(e={e_sig},i={i_sig})', user_q))

for idx, cat, typ, desc, why, q in cands:
    print(f'L{idx} [{cat}/{typ}] "{desc}"  <-{why}')
    print(f'     查詢: {q[:80]}')
