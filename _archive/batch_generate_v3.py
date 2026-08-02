import json
import random
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer.validate_json import extract_tool_call, validate_record

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

REFERENCE_DATE = date(2026, 7, 20)

def random_date(rng, start_delta=-30, end_delta=10):
    return REFERENCE_DATE + timedelta(days=rng.randint(start_delta, end_delta))

def date_to_relative_str(d, ref=REFERENCE_DATE):
    delta = (d - ref).days
    if delta == 0:
        return "今天"
    elif delta == -1:
        return "昨天"
    elif delta == -2:
        return "前天"
    elif delta == 1:
        return "明天"
    elif delta == 2:
        return "後天"
    elif delta < -1 and delta >= -7:
        weekdays = ["", "一", "二", "三", "四", "五", "六", "日"]
        ref_weekday = ref.weekday()
        return f"上週{weekdays[(d.weekday())]}"
    elif delta > 1 and delta <= 7:
        weekdays = ["", "一", "二", "三", "四", "五", "六", "日"]
        return f"下週{weekdays[(d.weekday())]}"
    elif delta <= -7 and delta >= -14:
        return "上上週"
    elif delta < -14 and delta >= -60:
        return "上個月"
    elif delta < -60:
        return "幾個月前"
    else:
        return d.strftime("%m月%d日")

def date_to_qualifier(d, ref=REFERENCE_DATE):
    delta = (d - ref).days
    if delta == 0:
        return ""
    elif delta == -1:
        return "昨天"
    elif delta == -2:
        return "前天"
    elif delta < 0:
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
        ref_wd = ref.weekday()
        d_wd = d.weekday()
        if d_wd < ref_wd:
            return f"上週{weekday_names[d_wd]}"
        else:
            return d.strftime("%m月%d日")
    else:
        weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
        d_wd = d.weekday()
        if d_wd > ref.weekday():
            return f"這週{weekday_names[d_wd]}"
        return d.strftime("%m月%d日")

def make_amounts(count, seed=42):
    rng = random.Random(seed)
    pool = list(range(10, 500001))
    rng.shuffle(pool)
    return pool[:count]

def make_sample(diff_level, cat, acc, amt, rtype, user_text, d):
    d_str = d.strftime("%Y-%m-%d")
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
    system_content = f"今天是 {REFERENCE_DATE.strftime('%Y-%m-%d')}。你是一個記帳助理。你被賦予了以下 tools:\n" + json.dumps(tool_def, ensure_ascii=False)
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

def gen(used, ei):
    while True:
        amt = AMOUNTS[ei[0]]
        ei[0] += 1
        if ei[0] >= len(AMOUNTS):
            ei[0] = 0
        return amt

AMOUNTS = make_amounts(100000)
ei = [0]

L1_TEXTS = dict(
    expense=[
        "{rel}{c}花了{amt}元用{a}",
        "{rel}在{c}消費{amt}元{a}",
        "{rel}{c}支出{amt}元{a}付款",
        "{rel}買{c}用了{amt}元{a}",
        "{rel}{c}{amt}元{a}結帳",
        "{rel}{c}{amt}元用{a}",
        "{rel}花{amt}元在{c}{a}",
    ],
    income=[
        "{rel}{c}{amt}元入{a}",
        "{rel}收到{c}{amt}元匯到{a}",
        "{rel}{c}{amt}元入帳{a}",
        "{rel}入帳{c}{amt}元{a}",
        "{rel}獲得{c}{amt}元已入{a}",
        "{rel}收到{c}{amt}元{a}",
    ]
)

L2_TEXTS = dict(
    expense=[
        "{rel}跟同事去{c}吃商業午餐花了{amt}元用{a}結帳",
        "{rel}下班去{c}買東西花了{amt}元用{a}",
        "{rel}跟朋友聚餐吃{c}一個人{amt}元用{a}刷卡",
        "{rel}叫外送{c}折價後{amt}元用{a}",
        "{rel}去{c}買早餐花{amt}元用{a}",
        "{rel}去{c}玩了一天花了{amt}元用{a}",
        "{rel}跟朋友唱歌一個人攤{amt}元用{a}",
        "{rel}逛{c}買東西花了{amt}元用{a}",
        "{rel}搭車花了一天交通費{amt}元",
        "{rel}下雨叫車花了{amt}元用{a}",
        "{rel}{a}儲值{amt}元",
        "{rel}去{c}補貨花了{amt}元用{a}",
        "{rel}去{c}買菜花了{amt}元用{a}",
        "{rel}買日用品花了{amt}元用{a}",
        "{rel}去{c}採購結帳{amt}元用{a}",
        "{rel}換季去{c}買衣服花了{amt}元用{a}",
        "{rel}週年慶買保養品花了{amt}元用{a}",
        "{rel}買3C配件{amt}元用{a}",
        "{rel}買遊戲花了{amt}元用{a}",
        "{rel}買寵物用品花了{amt}元用{a}",
        "{rel}帶寵物洗澡花了{amt}元用{a}",
        "{rel}去藥局買藥花了{amt}元用{a}",
        "{rel}看醫生花了{amt}元用{a}",
        "{rel}買線上課程{amt}元用{a}",
        "{rel}買書花了{amt}元用{a}",
        "{rel}去IKEA買家具花了{amt}元用{a}",
        "{rel}去{c}花了{amt}元用{a}",
        "{rel}{c}消費{amt}元帳單用{a}",
        "{rel}繳了{c}{amt}元用{a}",
        "{rel}{c}花了{amt}元{a}付款",
        "{rel}買{amt}元{c}用{a}",
        "{rel}{c}{amt}元已刷卡{a}",
        "{rel}{c}扣款{amt}元{a}",
        "{rel}{c}{a}{amt}元支出",
    ],
    income=[
        "{rel}發薪{amt}元入{a}",
        "{rel}獎金{amt}元匯到{a}",
        "{rel}收到{c}{amt}元入{a}",
        "{rel}{c}{amt}元進{a}",
        "{rel}退稅{amt}元匯到{a}",
        "{rel}房租收入{amt}元到{a}",
        "{rel}賣二手東西得{amt}元到{a}",
        "{rel}{c}獎金{amt}元入{a}",
        "{rel}接案完成收到{amt}元入{a}",
        "{rel}定存到期{amt}元入{a}",
        "{rel}{c}{amt}元匯入{a}",
        "{rel}{c}收入{amt}元{a}",
        "{rel}{a}收到{amt}元{c}",
    ]
)

L3_EXPENSE = [
    lambda rel,c,a,amt,ds: f"{rel}五天午餐平均一天{amt//5}元，總共{amt}元用{a}付的",
    lambda rel,c,a,amt,ds: f"{rel}跟朋友去吃{c}總帳單{amt}元我先刷卡，他們各轉{amt//4}元給我",
    lambda rel,c,a,amt,ds: f"{rel}去了三次{c}，三次加總{amt}元刷{a}",
    lambda rel,c,a,amt,ds: f"{rel}買衣服{amt//2}元，今天買鞋子{amt-amt//2}元，都用{a}同一天",
    lambda rel,c,a,amt,ds: f"{rel}電費{amt//3}元、水費{amt//3}元、瓦斯{amt-amt//3-amt//3}元都從{a}扣",
    lambda rel,c,a,amt,ds: f"{rel}繳了{c}帳單{amt}元用{a}",
    lambda rel,c,a,amt,ds: f"{rel}健身房入會加月費{amt}元用{a}",
    lambda rel,c,a,amt,ds: f"{rel}買設備{amt}元用{a}",
    lambda rel,c,a,amt,ds: f"{rel}去{c}花了{amt}元用{a}，總共三筆合計",
]
L3_INCOME = [
    lambda rel,c,a,amt,ds: f"{rel}本薪加津貼總共{amt}元入{a}",
    lambda rel,c,a,amt,ds: f"{rel}接案收到訂金{amt//3}元尾款{amt-amt//3}元總共{amt}元先記今天",
    lambda rel,c,a,amt,ds: f"{rel}獎金總共{amt}元入了{a}",
    lambda rel,c,a,amt,ds: f"{rel}配息總共{amt}元入{a}",
]

def generate_batch_v3(count_l1=1200, count_l2=1000, count_l3=800):
    rng = random.Random(42)
    samples = []
    used = set()

    def pick(cat, acc, amt, rtype):
        key = (amt, cat, acc, rtype)
        if key in used:
            return False
        used.add(key)
        return True

    for level, count, ttype, texts_fn, l3_fns in [
        ("Level-1 (Simple)", count_l1, None, L1_TEXTS, None),
        ("Level-2 (Noise)", count_l2, None, L2_TEXTS, None),
        ("Level-3 (Reasoning)", count_l3, ["expense","income"], None, [L3_EXPENSE, L3_INCOME]),
    ]:
        for _ in range(count):
            if ttype:
                rtype = rng.choice(ttype)
            else:
                rtype = rng.choices(["expense", "income"], weights=[0.55, 0.45])[0]
            
            if level == "Level-3 (Reasoning)":
                cat = rng.choice(CATEGORIES)
                acc = rng.choice(ACCOUNTS)
                amt = AMOUNTS[ei[0]]; ei[0] += 1
                if not pick(cat, acc, amt, rtype):
                    continue
                d = random_date(rng)
                rel = date_to_qualifier(d) if rng.random() < 0.7 else ""
                fn = rng.choice(l3_fns[0] if rtype == "expense" else l3_fns[1])
                text = fn(rel, cat, acc, amt, d.strftime("%Y-%m-%d"))
            else:
                texts = texts_fn[rtype]
                cat = rng.choice(CATEGORIES)
                acc = rng.choice(ACCOUNTS)
                amt = AMOUNTS[ei[0]]; ei[0] += 1
                if not pick(cat, acc, amt, rtype):
                    continue
                d = random_date(rng)
                rel = date_to_qualifier(d) if rng.random() < 0.7 else ""
                t = rng.choice(texts)
                text = t.format(rel=rel, c=cat, a=acc, amt=amt)
            
            samples.append(make_sample(level, cat, acc, amt, rtype, text, d))
    
    return samples

def main():
    raw_path = "dataset/raw_generated.jsonl"
    llm_path = "dataset/llm_generated.jsonl"

    print("Generating 3000 diverse samples with date variation...")
    samples = generate_batch_v3(count_l1=1200, count_l2=1000, count_l3=800)
    print(f"Generated {len(samples)} valid samples")

    llm_samples = []
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        llm_samples.append(json.loads(line))
                    except:
                        pass
        print(f"Loaded {len(llm_samples)} LLM samples")

    all_samples = samples + llm_samples
    rng = random.Random(123)
    rng.shuffle(all_samples)

    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    counts = {"Level-1 (Simple)": 0, "Level-2 (Noise)": 0, "Level-3 (Reasoning)": 0}
    inc = exp = 0
    date_set = set()
    for s in all_samples:
        d = s.get("difficulty_level", "")
        for k in counts:
            if d.startswith(k[:7]):
                counts[k] += 1; break
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        if '"type": "income"' in ast:
            inc += 1
        else:
            exp += 1
        try:
            args = extract_tool_call(ast)
            date_set.add(args.get("date", ""))
        except:
            pass

    passed = failed = 0
    for s in all_samples:
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        try:
            args = extract_tool_call(ast)
            valid, _ = validate_record(args)
            if valid: passed += 1
            else: failed += 1
        except:
            failed += 1

    amt_seen = set()
    dup_amts = 0
    for s in all_samples:
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        try:
            args = extract_tool_call(ast)
            a = args.get("amount", 0)
            if a in amt_seen: dup_amts += 1
            amt_seen.add(a)
        except:
            pass

    print(f"\nTotal: {len(all_samples)}")
    print(f"L1: {counts['Level-1 (Simple)']}, L2: {counts['Level-2 (Noise)']}, L3: {counts['Level-3 (Reasoning)']}")
    print(f"Income: {inc}, Expense: {exp}")
    print(f"Parser: {passed} passed, {failed} failed")
    print(f"Unique amounts: {len(amt_seen)}/{len(all_samples)} ({dup_amts} dups)")
    print(f"Unique dates: {len(date_set)}")

    rng.shuffle(all_samples)
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
