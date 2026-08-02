import json
import random
import sys
import os
import calendar
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer.validate_json import extract_tool_call, validate_record

REFERENCE_DATES = [date(2026, 7, 20)]

def _cn_num(n):
    cn = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
          "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
          "二十一", "二十二", "二十三", "二十四", "二十五", "二十六", "二十七", "二十八", "二十九", "三十",
          "三十一"]
    return cn[n]

_NOISE_CHARS = {"月": ["越", "曰"], "日": ["曰", "白"], "號": ["好", "毫"]}

def _apply_noise(text, p=0.15):
    if random.random() >= p:
        return text
    for orig, replacements in _NOISE_CHARS.items():
        if orig in text:
            text = text.replace(orig, random.choice(replacements), 1)
            break
    return text

def build_colloquial_prefix_map(ref_date):
    """Build a deterministic mapping of colloquial date expression -> date.
    ~156 variants, ~18 appearances each with 10000 total samples."""
    cpm = {}
    # 1. Day-only: Chinese + digit day-names 一號~三十一號 + 初一~初十 + 1號~31號
    for d in range(1, 32):
        try:
            actual = date(ref_date.year, ref_date.month, d)
        except ValueError:
            break
        cd = _cn_num(d)
        cpm[f"{cd}號"] = actual
        cpm[f"{d}號"] = actual
        if d <= 10:
            cpm[f"初{cd}"] = actual
    # 2. Month-day: ALL months, key days, formats: cm月cd日 + cm月cd號
    key_days = [1, 5, 10, 15, 20, 25]
    for m in range(1, 13):
        cm = _cn_num(m)
        md_limit = calendar.monthrange(ref_date.year, m)[1]
        for d in key_days:
            if d > md_limit:
                continue
            actual = date(ref_date.year, m, d)
            cd = _cn_num(d)
            cpm[f"{cm}月{cd}日"] = actual
            cpm[f"{cm}月{cd}號"] = actual
    # 3. Mix format: cm月d號 (Chinese month + digit day) for key months
    for m in [5, 7, 10, 12]:
        cm = _cn_num(m)
        for d in key_days:
            actual = date(ref_date.year, m, d)
            cpm[f"{cm}月{d}號"] = actual
    # 4. Compound numerals: month 7, days 11-31
    for d in range(11, 32):
        actual = date(ref_date.year, 7, d)
        cd = _cn_num(d)
        cpm[f"七月{cd}日"] = actual
        cpm[f"七月{cd}號"] = actual
    # 5. Digit formats: m月d日 + m月d號 + m/d for common dates
    for m, d in [(1, 1), (5, 1), (7, 5), (7, 15), (7, 20), (7, 25), (12, 25), (12, 31)]:
        actual = date(ref_date.year, m, d)
        cpm[f"{m}月{d}日"] = actual
        cpm[f"{m}月{d}號"] = actual
        cpm[f"{m}/{d}"] = actual
    # 6. Month-only: ALL months with Chinese and digit names
    for m in range(1, 13):
        cpm[f"{_cn_num(m)}月"] = date(ref_date.year, m, 1)
        cpm[f"{m}月"] = date(ref_date.year, m, 1)
    # 7. Explicit composition-gap cases
    for m, d in [(11, 18), (12, 25), (12, 31)]:
        actual = date(ref_date.year, m, d)
        cm = _cn_num(m)
        cd = _cn_num(d)
        cpm[f"{cm}月{cd}日"] = actual
        cpm[f"{cm}月{d}日"] = actual
        cpm[f"{m}月{cd}日"] = actual
    return cpm

def build_prefix_map(ref_date):
    """Build deterministic prefix->date mapping for a given ref_date.
    Each prefix maps to exactly ONE date for this ref_date.
    Covers all common relative date expressions."""
    wd = ["一", "二", "三", "四", "五", "六", "日"]
    pm = {}
    ref_monday = ref_date - timedelta(days=ref_date.weekday())

    # Simple day offsets
    pm["今天"] = ref_date
    pm["明天"] = ref_date + timedelta(days=1)
    pm["後天"] = ref_date + timedelta(days=2)
    pm["大後天"] = ref_date + timedelta(days=3)
    pm["昨天"] = ref_date - timedelta(days=1)
    pm["前天"] = ref_date - timedelta(days=2)
    pm["大前天"] = ref_date - timedelta(days=3)

    # This week
    for i in range(7):
        d = ref_monday + timedelta(days=i)
        if d != ref_date:
            pm[f"這週{wd[i]}"] = d

    # Last week
    for i in range(7):
        pm[f"上週{wd[i]}"] = ref_monday - timedelta(days=7 - i)

    # Next week
    for i in range(7):
        pm[f"下週{wd[i]}"] = ref_monday + timedelta(days=7 + i)

    # Two-week offsets
    pm["上上週"] = ref_monday - timedelta(days=14)
    pm["下下週"] = ref_monday + timedelta(days=14)

    # Month offsets
    pm["這個月"] = ref_date
    for months in [1, 2, 3]:
        m = ref_date.month - months
        y = ref_date.year
        while m < 1:
            m += 12
            y -= 1
        last_day = calendar.monthrange(y, m)[1]
        day = min(ref_date.day, last_day)
        d = date(y, m, day)
        if months == 1:
            pm["上個月"] = d
        elif months == 2:
            pm["上上個月"] = d
        else:
            pm[f"{months}個月前"] = d

    for months in [1, 2, 3]:
        m = ref_date.month + months
        y = ref_date.year
        while m > 12:
            m -= 12
            y += 1
        last_day = calendar.monthrange(y, m)[1]
        day = min(ref_date.day, last_day)
        d = date(y, m, day)
        if months == 1:
            pm["下個月"] = d
        elif months == 2:
            pm["下下個月"] = d
        else:
            pm[f"{months}個月後"] = d

    # Year offsets
    for years in [1, 2]:
        d = date(ref_date.year - years, ref_date.month, ref_date.day)
        if years == 1:
            pm["去年"] = d
        else:
            pm["前年"] = d

    for years in [1, 2]:
        d = date(ref_date.year + years, ref_date.month, ref_date.day)
        if years == 1:
            pm["明年"] = d
        else:
            pm["後年"] = d

    return pm

CATEGORIES = [
    "餐飲飲食", "休閒娛樂", "交通出行", "生活繳費", "醫療保健",
    "教育學習", "日常雜貨", "服飾美妝", "數位服務", "投資理財",
    "薪資收入", "獎金紅利", "副業外快", "寵物支出", "房租房貸",
    "保險費用", "人情往來", "家居裝修", "3C電子", "運動健身"
]

ACCOUNTS = [
    "現金", "信用卡", "悠遊卡", "一卡通", "街口支付",
    "LINE Pay", "Apple Pay", "Google Pay", "郵局帳戶",
    "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"
]

B_CLASS_EXPENSE_TEXTS = [
    ("餐飲飲食", lambda m,a: f"去餐廳吃飯花了{m}元用{a}"),
    ("餐飲飲食", lambda m,a: f"買了杯咖啡{m}元用{a}"),
    ("餐飲飲食", lambda m,a: f"叫了外送{m}元用{a}"),
    ("餐飲飲食", lambda m,a: f"朋友聚餐{m}元用{a}"),
    ("餐飲飲食", lambda m,a: f"買便當{m}元用{a}"),
    ("交通出行", lambda m,a: f"搭計程車花了{m}元用{a}"),
    ("交通出行", lambda m,a: f"加油{m}元用{a}"),
    ("交通出行", lambda m,a: f"坐捷運{m}元用{a}"),
    ("交通出行", lambda m,a: f"停車費{m}元用{a}"),
    ("3C電子", lambda m,a: f"買了一張顯卡{m}元用{a}"),
    ("3C電子", lambda m,a: f"買了一條充電線{m}元用{a}"),
    ("3C電子", lambda m,a: f"換了新手機{m}元用{a}"),
    ("3C電子", lambda m,a: f"買了滑鼠{m}元用{a}"),
    ("保險費用", lambda m,a: f"繳了車險{m}元用{a}"),
    ("保險費用", lambda m,a: f"付了醫療險{m}元用{a}"),
    ("保險費用", lambda m,a: f"繳了保險費{m}元用{a}"),
    ("日常雜貨", lambda m,a: f"買菜{m}元用{a}"),
    ("日常雜貨", lambda m,a: f"超市購物{m}元用{a}"),
    ("日常雜貨", lambda m,a: f"買了牛奶麵包{m}元用{a}"),
    ("休閒娛樂", lambda m,a: f"看電影{m}元用{a}"),
    ("休閒娛樂", lambda m,a: f"買了遊戲{m}元用{a}"),
    ("休閒娛樂", lambda m,a: f"唱歌{m}元用{a}"),
    ("醫療保健", lambda m,a: f"看醫生掛號費{m}元用{a}"),
    ("醫療保健", lambda m,a: f"買藥{m}元用{a}"),
    ("醫療保健", lambda m,a: f"牙醫{m}元用{a}"),
    ("教育學習", lambda m,a: f"買書{m}元用{a}"),
    ("教育學習", lambda m,a: f"繳學費{m}元用{a}"),
    ("教育學習", lambda m,a: f"報名課程{m}元用{a}"),
    ("寵物支出", lambda m,a: f"買飼料{m}元用{a}"),
    ("寵物支出", lambda m,a: f"寵物看醫生{m}元用{a}"),
    ("寵物支出", lambda m,a: f"買貓砂{m}元用{a}"),
    ("服飾美妝", lambda m,a: f"買衣服{m}元用{a}"),
    ("服飾美妝", lambda m,a: f"買化妝品{m}元用{a}"),
    ("服飾美妝", lambda m,a: f"買鞋子{m}元用{a}"),
    ("生活繳費", lambda m,a: f"繳電費{m}元用{a}"),
    ("生活繳費", lambda m,a: f"繳水費{m}元用{a}"),
    ("生活繳費", lambda m,a: f"繳網路費{m}元用{a}"),
    ("運動健身", lambda m,a: f"去健身房{m}元用{a}"),
    ("運動健身", lambda m,a: f"買運動用品{m}元用{a}"),
    ("人情往來", lambda m,a: f"包紅包{m}元用{a}"),
    ("人情往來", lambda m,a: f"送禮{m}元用{a}"),
    ("數位服務", lambda m,a: f"繳月費{m}元用{a}"),
    ("數位服務", lambda m,a: f"買雲端空間{m}元用{a}"),
]
B_CLASS_INCOME_TEXTS = [
    ("薪資收入", lambda m,a: f"公司發薪水{m}元用{a}"),
    ("薪資收入", lambda m,a: f"月薪{m}元入帳{a}"),
    ("獎金紅利", lambda m,a: f"年終獎金{m}元用{a}轉入"),
    ("獎金紅利", lambda m,a: f"分紅{m}元用{a}"),
    ("副業外快", lambda m,a: f"接案收入{m}元用{a}"),
    ("副業外快", lambda m,a: f"家教薪水{m}元用{a}"),
    ("副業外快", lambda m,a: f"外包案{m}元用{a}"),
]

def make_amounts(exp_count, inc_count, seed=42):
    rng = random.Random(seed)
    exp_pool = list(range(15, 100001))
    inc_pool = list(range(500, 500001))
    rng.shuffle(exp_pool)
    rng.shuffle(inc_pool)
    return exp_pool[:exp_count], inc_pool[:inc_count]

def make_sample(diff_level, cat, acc, amt, rtype, user_text, d, ref_date):
    d_str = d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)
    sub_cats = random.sample(CATEGORIES, k=random.randint(4, 6))
    if cat not in sub_cats:
        sub_cats[-1] = cat
    sub_accs = random.sample(ACCOUNTS, k=random.randint(3, 5))
    if acc not in sub_accs:
        sub_accs[-1] = acc
    tool_def = {
        "name": "add_record",
        "description": "新增一筆記帳記錄",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": sub_cats},
                "account": {"type": "string", "enum": sub_accs},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": ["expense", "income"]},
                "date": {"type": "string", "description": "ISO 8601 格式日期，例如 YYYY-MM-DD"}
            },
            "required": ["amount", "category", "account", "type", "date"]
        }
    }
    system_content = f"今天是 {ref_date.strftime('%Y-%m-%d')}。你是一個記帳助理。你被賦予了以下 tools:\n" + json.dumps(tool_def, ensure_ascii=False)
    args_dict = {"amount": amt, "category": cat, "account": acc, "description": f"{cat}{rtype}", "type": rtype, "date": d_str}
    inner = json.dumps({"name": "add_record", "args": args_dict}, ensure_ascii=True)
    assistant_content = "<tool_call>" + inner + "</tool_call>"
    return {
        "difficulty_level": diff_level,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_batch_v2(count_l1=6000, count_l2=4000, count_l3=3000):
    samples = []
    used = set()
    exp_amts, inc_amts = make_amounts(20000, 10000)
    ei = ii = 0

    def next_amt(rtype):
        nonlocal ei, ii
        if rtype == "expense":
            v = exp_amts[ei]
            ei += 1
            return v
        else:
            v = inc_amts[ii]
            ii += 1
            return v

    def pick(diff, cat, acc, amt, rtype):
        key = (amt, cat, acc, rtype)
        if key in used:
            return False
        used.add(key)
        return True

    L1_EXPENSE_TEXTS = [
        lambda c,a,m: f"今天{c}消費了{m}元用{a}結帳",
        lambda c,a,m: f"去{c}買東西花了{m}元用{a}付款",
        lambda c,a,m: f"{m}元在{c}用{a}支付",
        lambda c,a,m: f"在{c}花了{m}元，用{a}",
        lambda c,a,m: f"買{c}用了{m}元，{a}結帳",
        lambda c,a,m: f"剛剛{c}消費{m}元，{a}",
        lambda c,a,m: f"今天{c}支出{m}元，{a}付清",
        lambda c,a,m: f"{c}買{m}元用{a}",
        lambda c,a,m: f"用{a}付了{c}的{m}元",
        lambda c,a,m: f"支出{m}元在{c}，{a}付款",
        lambda c,a,m: f"今天去{c}花了{m}元刷{a}",
        lambda c,a,m: f"用{a}繳了{m}元{c}費用",
        lambda c,a,m: f"{c}花了{m}元啦用{a}",
        lambda c,a,m: f"刷{a}付{c}{m}元",
        lambda c,a,m: f"今天{c}付款{m}元{a}搞定",
    ]
    L1_INCOME_TEXTS = [
        lambda c,a,m: f"收到{c}{m}元匯到{a}",
        lambda c,a,m: f"{c}{m}元入{a}帳戶",
        lambda c,a,m: f"{c}{m}元入帳到{a}",
        lambda c,a,m: f"{a}收到{c}{m}元",
        lambda c,a,m: f"{c}收入{m}元已入{a}",
        lambda c,a,m: f"今天{a}入帳{c}{m}元",
        lambda c,a,m: f"{c}進帳{m}元到{a}",
        lambda c,a,m: f"收到{c}的{m}元匯入{a}",
        lambda c,a,m: f"{a}入金{c}{m}元",
    ]

    L2_EXPENSE_TEXTS = [
        lambda c,a,m: f"今天中午跟同事去{c}吃了商業午餐，花了{m}元用{a}結帳",
        lambda c,a,m: f"下班後去{c}買東西，總共{m}元用{a}付的",
        lambda c,a,m: f"跟朋友約吃{c}，一個人{m}元用{a}刷卡",
        lambda c,a,m: f"叫了外送點{c}，折價後{m}元用{a}付款",
        lambda c,a,m: f"早上去{c}買早餐花{m}元用{a}嗶一下",
        lambda c,a,m: f"週末去{c}玩了一天花了{m}元用{a}",
        lambda c,a,m: f"跟朋友去唱KTV一個人分攤{m}元用{a}結帳",
        lambda c,a,m: f"Netflix月費{m}元刷{a}",
        lambda c,a,m: f"逛{c}不小心買了東西花了{m}元用{a}",
        lambda c,a,m: f"搭{a}上下班一天交通費{m}元",
        lambda c,a,m: f"下雨天叫車花了{m}元用{a}付款",
        lambda c,a,m: f"{a}儲值了{m}元這週通勤用",
        lambda c,a,m: f"去{c}補貨花了{m}元用{a}付的",
        lambda c,a,m: f"週六去{c}買菜花了{m}元用{a}結帳",
        lambda c,a,m: f"補了一批日用品花了{m}元刷{a}",
        lambda c,a,m: f"去{c}補貨一大車結帳{m}元用{a}",
        lambda c,a,m: f"換季去{c}買衣服花了{m}元用{a}刷",
        lambda c,a,m: f"週年慶買了一組保養品花了{m}元刷{a}",
        lambda c,a,m: f"路過{c}買了外套花了{m}元用{a}",
        lambda c,a,m: f"買了3C配件{m}元用{a}付款",
        lambda c,a,m: f"Steam特賣買遊戲花了{m}元刷{a}",
        lambda c,a,m: f"Spotify家庭方案我主揪一人{m}元用{a}",
        lambda c,a,m: f"買貓罐頭花了{m}元用{a}結帳",
        lambda c,a,m: f"帶狗狗洗澡剪毛花了{m}元用{a}",
        lambda c,a,m: f"去藥局買藥花了{m}元用{a}付",
        lambda c,a,m: f"看醫生掛號加療程{m}元刷{a}",
        lambda c,a,m: f"買線上課程特價{m}元用{a}",
        lambda c,a,m: f"去書局買書花了{m}元用{a}結帳",
        lambda c,a,m: f"去IKEA買收納櫃花了{m}元用{a}",
        lambda c,a,m: f"買檯燈花了{m}元用{a}結帳",
        # More colloquial variants
        lambda c,a,m: f"肚子餓去{c}吃東西花了{m}元直接用{a}付",
        lambda c,a,m: f"超商補貨買了些{c}總共{m}元用{a}嗶一下",
        lambda c,a,m: f"下班去{c}逛了一圈不小心花了{m}元刷{a}",
        lambda c,a,m: f"趁特價在{c}買了{m}元的東西用{a}結帳好划算",
        lambda c,a,m: f"去{c}買了{m}元的必需品用{a}付的帳",
        lambda c,a,m: f"跟同事揪團買{c}一人{m}元用{a}轉帳給他",
        lambda c,a,m: f"叫熊貓外送點{c}用了{m}元用{a}線上付款",
        lambda c,a,m: f"去{c}加油花了{m}元刷{a}",
        lambda c,a,m: f"帶家人去{c}吃飯總共{m}元我買單用{a}",
        lambda c,a,m: f"嘴饞去{c}買零食花了{m}元用{a}結帳",
        lambda c,a,m: f"在{c}買了{m}元的日用品用{a}付款",
        lambda c,a,m: f"繳{c}的帳單{m}元用{a}扣款",
        lambda c,a,m: f"去{c}買了{m}元的用品{a}付款",
        lambda c,a,m: f"今天在{c}噴了{m}元用{a}刷卡心痛",
        lambda c,a,m: f"巷口{c}買了{m}元直接{a}結帳",
        lambda c,a,m: f"跟朋友去{c}聚餐一個人分攤{m}元用{a}",
        lambda c,a,m: f"忘記帶錢包用{a}付了{c}{m}元",
        lambda c,a,m: f"{c}的{m}元帳單用{a}付清了",
        lambda c,a,m: f"在路上看到{c}就進去買了{m}元用{a}",
        lambda c,a,m: f"{a}扣款{c}{m}元成功",
    ]
    _B_CLASS_EXP = B_CLASS_EXPENSE_TEXTS
    _B_CLASS_INC = B_CLASS_INCOME_TEXTS

    L2_INCOME_TEXTS = [
        lambda c,a,m: f"發薪日！{c}{m}元入{a}",
        lambda c,a,m: f"年終獎金{c}{m}元匯到{a}",
        lambda c,a,m: f"接外包收到{c}{m}元入{a}",
        lambda c,a,m: f"股票配息{c}{m}元進{a}",
        lambda c,a,m: f"收到{c}退稅{m}元匯到{a}",
        lambda c,a,m: f"房租收入{m}元匯到{a}",
        lambda c,a,m: f"賣二手東西賣了{m}元匯到{a}",
        lambda c,a,m: f"公司績效獎金{m}元入{a}",
        lambda c,a,m: f"接翻譯完成收到{c}{m}元入{a}",
        lambda c,a,m: f"美金定存到期{m}元入外幣帳戶",
        lambda c,a,m: f"{c}的款項{m}元進帳到{a}",
        lambda c,a,m: f"收到{c}的{m}元匯款存進{a}",
        lambda c,a,m: f"公司發{c}{m}元直接入{a}戶頭",
        lambda c,a,m: f"客戶匯款{c}{m}元到{a}",
        lambda c,a,m: f"投資收益{m}元入{a}帳戶",
    ]

    # Pre-build prefix/colloquial maps for all reference dates
    _prefix_maps = {rd: build_prefix_map(rd) for rd in REFERENCE_DATES}
    _colloquial_maps = {rd: build_colloquial_prefix_map(rd) for rd in REFERENCE_DATES}

    def _resolve_date(ref_date):
        pm = _prefix_maps[ref_date]
        pkl = list(pm.keys())
        cm = _colloquial_maps[ref_date]
        ckl = list(cm.keys())
        dr = random.random()
        if dr < 0.55:
            pref = random.choice(pkl)
            return pm[pref], pref
        elif dr < 0.90:
            pref = random.choice(ckl)
            return cm[pref], pref
        return ref_date, ""

    def _apply_prefix(text, pref):
        if not pref:
            return text
        if "今天" in text:
            return text.replace("今天", pref, 1)
        if not text.startswith(pref):
            return pref + text
        return text

    def _make_b_sample(rtype, amt):
        templates = _B_CLASS_EXP if rtype == "expense" else _B_CLASS_INC
        cat, fn = random.choice(templates)
        acc = random.choice(ACCOUNTS)
        return cat, acc, fn(amt, acc)

    # L1: simple
    for _ in range(count_l1):
        ref_date = random.choice(REFERENCE_DATES)
        d, pref = _resolve_date(ref_date)
        rtype = "expense" if random.random() < 0.55 else "income"
        amt = next_amt(rtype)
        tc = random.random()
        if tc < 0.6:
            cat = random.choice(CATEGORIES)
            acc = random.choice(ACCOUNTS)
            if not pick("L1", cat, acc, amt, rtype):
                continue
            if rtype == "expense":
                text = random.choice(L1_EXPENSE_TEXTS)(cat, acc, amt)
            else:
                text = random.choice(L1_INCOME_TEXTS)(cat, acc, amt)
        elif tc < 0.9:
            cat, acc, text = _make_b_sample(rtype, amt)
            if not pick("L1", cat, acc, amt, rtype):
                continue
        else:
            cat, acc, text = _make_b_sample(rtype, amt)
            if not pick("L1", cat, acc, amt, rtype):
                continue
            if not pref or random.random() < 0.5:
                cm = _colloquial_maps[ref_date]
                ckl = list(cm.keys())
                pref = random.choice(ckl)
                d = cm[pref]
        text = _apply_prefix(text, pref)
        samples.append(make_sample("Level-1 (Simple)", cat, acc, amt, rtype, text, d, ref_date))

    # L2: noisy
    for _ in range(count_l2):
        ref_date = random.choice(REFERENCE_DATES)
        d, pref = _resolve_date(ref_date)
        rtype = "expense" if random.random() < 0.65 else "income"
        amt = next_amt(rtype)
        tc = random.random()
        if tc < 0.6:
            cat = random.choice(CATEGORIES)
            acc = random.choice(ACCOUNTS)
            if not pick("L2", cat, acc, amt, rtype):
                continue
            if rtype == "expense":
                text = random.choice(L2_EXPENSE_TEXTS)(cat, acc, amt)
            else:
                text = random.choice(L2_INCOME_TEXTS)(cat, acc, amt)
        elif tc < 0.9:
            cat, acc, text = _make_b_sample(rtype, amt)
            if not pick("L2", cat, acc, amt, rtype):
                continue
        else:
            cat, acc, text = _make_b_sample(rtype, amt)
            if not pick("L2", cat, acc, amt, rtype):
                continue
            if not pref or random.random() < 0.5:
                cm = _colloquial_maps[ref_date]
                ckl = list(cm.keys())
                pref = random.choice(ckl)
                d = cm[pref]
        text = _apply_prefix(text, pref)
        samples.append(make_sample("Level-2 (Noise)", cat, acc, amt, rtype, text, d, ref_date))

    # L3: reasoning / math
    for _ in range(count_l3):
        ref_date = random.choice(REFERENCE_DATES)
        d, pref = _resolve_date(ref_date)
        rtype = "expense" if random.random() < 0.6 else "income"
        amt = next_amt(rtype)
        tc = random.random()
        if tc < 0.7:
            cat = random.choice(CATEGORIES)
            acc = random.choice(ACCOUNTS)
            if not pick("L3", cat, acc, amt, rtype):
                continue
        else:
            cat, acc, _ = _make_b_sample(rtype, amt)
            if not pick("L3", cat, acc, amt, rtype):
                continue

        if rtype == "expense":
            sub = random.randint(1, amt // 3)
            r = random.randint(0, 5)
            if r == 0:
                text = f"五天午餐平均一天{sub}元，四天午餐總共{amt}元用{acc}付的"
                if pref:
                    text = pref + text
                else:
                    d = ref_date
            elif r == 1:
                n = random.randint(2, 5)
                text = f"跟{n-1}個朋友去吃{cat}總帳單{amt}元我先刷卡付了，他們各轉{amt//n}元給我"
                if pref:
                    text = pref + text
            elif r == 2:
                text = f"去了三次{cat}，三次加總{amt}元刷同一張{acc}"
                if pref:
                    text = pref + text
                else:
                    d = ref_date
            elif r == 3:
                a = amt // 2
                b = amt - a
                text = f"買衣服{a}元，買鞋子{b}元，都用{acc}記同一天"
                if pref:
                    text = pref + text
            elif r == 4:
                per = amt // 22
                text = f"每天搭公車一趟{per//2}元來回{per}元，上班22天總共{amt}元"
                d = ref_date
            else:
                a = amt // 3
                b = amt // 3
                c1 = amt - a - b
                text = f"電費{a}元、水費{b}元、瓦斯{c1}元，都從{acc}扣加總{amt}元"
                if pref:
                    text = pref + text
        else:
            r = random.randint(0, 3)
            if r == 0:
                text = f"本薪加津貼總共{amt}元入{acc}"
                if pref:
                    text = pref + text
            elif r == 1:
                text = f"接案子收到訂金{amt//3}元，尾款{amt-amt//3}元，總共{amt}元先記錄"
                if pref:
                    text = pref + text
            elif r == 2:
                text = f"獎金總共{amt}元入了{acc}"
                if pref:
                    text = pref + text
            else:
                text = f"股票配息加基金配息總共{amt}元入{acc}"
                if pref:
                    text = pref + text

        samples.append(make_sample("Level-3 (Reasoning)", cat, acc, amt, rtype, text, d, ref_date))

    return samples

def main():
    raw_path = "dataset/raw_generated.jsonl"
    llm_path = "dataset/llm_generated.jsonl"

    # Generate fresh
    print("Generating 15000 diverse samples with date variation...")
    samples = generate_batch_v2(count_l1=6000, count_l2=4000, count_l3=5000)
    print(f"Generated {len(samples)} valid samples")

    # Load any existing LLM data
    llm_samples = []
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        llm_samples.append(json.loads(line))
                    except:
                        pass
        print(f"Loaded {len(llm_samples)} LLM-generated samples")

    # Shuffle and write
    all_samples = samples + llm_samples
    random.shuffle(all_samples)

    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Count breakdown
    counts = {"Level-1 (Simple)": 0, "Level-2 (Noise)": 0, "Level-3 (Reasoning)": 0}
    inc = exp = 0
    for s in all_samples:
        d = s.get("difficulty_level", "")
        for k in counts:
            if d.startswith(k[:7]):
                counts[k] += 1
                break
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        if '"type": "income"' in ast:
            inc += 1
        else:
            exp += 1

    print(f"\nTotal: {len(all_samples)}")
    print(f"L1: {counts['Level-1 (Simple)']}, L2: {counts['Level-2 (Noise)']}, L3: {counts['Level-3 (Reasoning)']}")
    print(f"Income: {inc}, Expense: {exp}")

    # Validate
    passed = failed = 0
    for s in all_samples:
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        try:
            args = extract_tool_call(ast)
            valid, _ = validate_record(args)
            if valid:
                passed += 1
            else:
                failed += 1
        except:
            failed += 1
    print(f"Parser: {passed} passed, {failed} failed")

    # Check amount uniqueness
    amt_seen = set()
    dup_amts = 0
    for s in all_samples:
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        try:
            args = extract_tool_call(ast)
            a = args.get("amount", 0)
            if a in amt_seen:
                dup_amts += 1
            amt_seen.add(a)
        except:
            pass
    print(f"Unique amounts: {len(amt_seen)}/{len(all_samples)} ({dup_amts} duplicates)")

    # Train/test split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train = all_samples[:split_idx]
    test = all_samples[split_idx:]
    with open("dataset/train_strata.jsonl", "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open("dataset/test_strata.jsonl", "w", encoding="utf-8") as f:
        for s in test:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\ntrain_strata.jsonl: {len(train)} samples")
    print(f"test_strata.jsonl: {len(test)} samples")

if __name__ == "__main__":
    main()
