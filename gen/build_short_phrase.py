"""
gen/build_short_phrase.py — 短語式記帳樣本產生器（v4）

目標：修復「早餐60」型短語（物品+金額、無動詞、無帳戶線索）分類/金額錯誤率高的問題。

產出（與 dataset/v3/train_v3.jsonl 位元格式一致）：
- dataset/v4/short_train.jsonl   訓練用短語樣本
- dataset/v4/short_val.jsonl     驗證用（checkpoint selection 混入）
- dataset/v4/short_test.jsonl    測試用（物品詞與 train 零重疊；含使用者案例「早餐60」）

格式契約：
- system   : build_system_prompt(['add_record'], ref_date, enums, weekday=True)
- assistant:  + json.dumps({"name": "add_record", "args": {...}},
             ensure_ascii=False) + 
- args key 順序: amount, category, account, description, type, date
"""
import sys
import os
import json
import random
from datetime import date, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from build_prompt import build_system_prompt

TOOL_CALL_OPEN = '<|tool_call|>'
TOOL_CALL_CLOSE = '<|tool_result|>'

BASE_CATEGORIES = ["飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款", "薪水", "獎金",
                   "零用錢", "兼職", "投資", "利息", "欠款回收", "其他"]
EXTRA_CATEGORIES = ["咖啡", "停車費", "禮物", "油費", "美髮", "超商", "健身房", "訂閱",
                    "清潔用品", "書籍", "文具", "寵物", "保險", "水電費", "管理費"]
BASE_ACCOUNTS = ["現金", "信用卡", "悠遊卡", "一卡通", "街口支付", "LINE Pay", "Apple Pay",
                 "Google Pay", "郵局帳戶", "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"]
EXTRA_ACCOUNTS = ["玉山帳戶", "合庫帳戶", "歐付寶", "永豐信用卡", "富邦信用卡", "台新信用卡"]

# (物品詞, 分類, 金額下限, 金額上限, 收支類型)
EXPENSE_ITEMS = [
    # 飲食
    ("早餐", "飲食", 25, 120), ("午餐", "飲食", 50, 250), ("晚餐", "飲食", 80, 450),
    ("宵夜", "飲食", 40, 200), ("便當", "飲食", 45, 110), ("牛肉麵", "飲食", 80, 160),
    ("滷肉飯", "飲食", 30, 70), ("咖啡", "飲食", 45, 180), ("手搖飲", "飲食", 25, 75),
    ("珍奶", "飲食", 30, 70), ("飲料", "飲食", 20, 80), ("火鍋", "飲食", 300, 900),
    ("壽司", "飲食", 100, 400), ("拉麵", "飲食", 150, 400), ("漢堡", "飲食", 60, 180),
    ("披薩", "飲食", 100, 500), ("麵包", "飲食", 20, 90), ("自助餐", "飲食", 55, 90),
    ("雞排", "飲食", 40, 90), ("早午餐", "飲食", 150, 400), ("麵線", "飲食", 40, 90),
    ("水餃", "飲食", 50, 120), ("牛排", "飲食", 300, 1200), ("甜品", "飲食", 60, 200),
    ("早餐店", "飲食", 30, 100),
    # 日常
    ("衛生紙", "日常", 80, 350), ("洗衣精", "日常", 120, 400), ("洗髮精", "日常", 100, 350),
    ("牙膏", "日常", 50, 180), ("沐浴乳", "日常", 90, 300), ("垃圾袋", "日常", 30, 120),
    ("電池", "日常", 40, 150), ("毛巾", "日常", 60, 250), ("拖鞋", "日常", 80, 300),
    ("肥皂", "日常", 25, 80), ("日用品", "日常", 50, 500), ("清潔劑", "日常", 60, 220),
    ("襪子", "日常", 50, 200), ("內衣", "日常", 200, 800), ("雨傘", "日常", 80, 300),
    # 交通
    ("捷運", "交通", 15, 80), ("公車", "交通", 12, 60), ("高鐵", "交通", 350, 1600),
    ("YouBike", "交通", 10, 40), ("計程車", "交通", 70, 600), ("停車費", "交通", 20, 200),
    ("加油", "交通", 300, 1200), ("油錢", "交通", 200, 900), ("機票", "交通", 1500, 8000),
    ("火車票", "交通", 80, 500), ("UBike", "交通", 10, 40), ("Call車", "交通", 70, 500),
    # 娛樂
    ("電影票", "娛樂", 150, 350), ("KTV", "娛樂", 200, 900), ("遊戲點數", "娛樂", 100, 1500),
    ("網費", "娛樂", 300, 700), ("串流訂閱", "娛樂", 100, 400), ("演唱會門票", "娛樂", 1800, 6500),
    ("桌遊", "娛樂", 200, 1200), ("密室逃脫", "娛樂", 450, 900), ("電玩", "娛樂", 300, 2000),
    ("遊樂園", "娛樂", 400, 1800), ("唱K", "娛樂", 200, 800),
    # 醫療
    ("掛號", "醫療", 50, 300), ("掛號費", "醫療", 50, 300), ("感冒藥", "醫療", 80, 300),
    ("牙醫", "醫療", 300, 3000), ("健檢", "醫療", 1500, 6000), ("維他命", "醫療", 200, 900),
    ("皮膚科", "醫療", 200, 1500), ("腸胃藥", "醫療", 80, 260), ("疫苗", "醫療", 500, 3500),
    # 教育
    ("補習費", "教育", 2000, 8000), ("教科書", "教育", 150, 800), ("線上課程", "教育", 500, 3500),
    ("才藝班", "教育", 1200, 6000), ("書籍", "教育", 100, 600), ("文具", "教育", 30, 300),
    ("學費", "教育", 3000, 20000), ("考試報名費", "教育", 300, 2500),
    # 還款（expense）
    ("卡費", "還款", 500, 15000, "expense"),
    ("房貸", "還款", 8000, 30000, "expense"), ("車貸", "還款", 5000, 18000, "expense"),
    ("房貸繳款", "還款", 8000, 30000, "expense"),
    # 其他（expense）
    ("捐款", "其他", 100, 2000), ("罰單", "其他", 300, 1500), ("罰款", "其他", 300, 1500),
    ("禮金", "其他", 600, 6600), ("紅包", "其他", 600, 6600), ("隨禮", "其他", 600, 3600),
    ("水費", "其他", 200, 900), ("電費", "其他", 500, 3000), ("瓦斯費", "其他", 300, 1500),
    ("管理費", "其他", 500, 3000), ("網路費", "其他", 300, 800), ("訂閱費", "其他", 100, 500),
    ("保險費", "其他", 800, 6000), ("稅金", "其他", 1000, 20000),
]
# 統一補上 type=expense（上面個別標 income/expense 的 tuple 例外）
_norm_items = []
for tup in EXPENSE_ITEMS:
    if len(tup) == 4:
        item, cat, lo, hi = tup
        tup = (item, cat, lo, hi, "expense")
    _norm_items.append(tup)
EXPENSE_ITEMS = _norm_items

INCOME_ITEMS = [
    ("薪水", "薪水", 30000, 90000, "income"), ("月薪", "薪水", 30000, 90000, "income"),
    ("薪資", "薪水", 30000, 90000, "income"),
    ("獎金", "獎金", 2000, 50000, "income"), ("年終", "獎金", 10000, 100000, "income"),
    ("績效", "獎金", 2000, 30000, "income"), ("業績獎金", "獎金", 5000, 60000, "income"),
    ("零用錢", "零用錢", 300, 3000, "income"), ("兼職", "兼職", 500, 15000, "income"),
    ("接案", "兼職", 1000, 30000, "income"), ("家教", "兼職", 500, 3000, "income"),
    ("業配", "兼職", 2000, 30000, "income"),
    ("股息", "投資", 500, 20000, "income"), ("股票獲利", "投資", 1000, 50000, "income"),
    ("投資收益", "投資", 500, 30000, "income"), ("獲利了結", "投資", 1000, 60000, "income"),
    ("定存利息", "利息", 50, 2000, "income"), ("銀行利息", "利息", 30, 1500, "income"),
    ("利息", "利息", 30, 1500, "income"),
    ("朋友還錢", "欠款回收", 200, 10000, "income"), ("借款回收", "欠款回收", 500, 20000, "income"),
    ("還款入帳", "欠款回收", 500, 15000, "income"), ("討回欠款", "欠款回收", 300, 8000, "income"),
]

ALL_ITEMS = list(EXPENSE_ITEMS) + list(INCOME_ITEMS)

# 帳戶線索後綴 → (account, 是否寫進 description)
ACCOUNT_SUFFIX = [
    ("現金", "現金"), ("刷卡", "信用卡"), ("信用卡", "信用卡"), ("悠遊卡", "悠遊卡"),
    ("一卡通", "一卡通"), ("LINE Pay", "LINE Pay"), ("Apple Pay", "Apple Pay"),
    ("街口", "街口支付"), ("icash", "icash"), ("悠遊付", "悠遊付"),
]

# 日期前綴 → 天數偏移
DATE_PREFIX = [("昨天", -1), ("前天", -2), ("上週", -7)]


def _fmt_amount(rng, amount):
    """把金額渲染成用户可能的書寫形式，回傳 (書寫字串)。"""
    style = rng.random()
    if amount >= 10000 and style < 0.35:
        w = amount / 10000
        wtxt = f"{w:g}"
        return f"{wtxt}萬"
    if amount >= 10000 and style < 0.5:
        return f"{amount // 1000}k"
    if amount >= 1000 and style < 0.25:
        return f"{amount // 1000}千"
    if style < 0.18:
        return f"{amount}元"
    if style < 0.3:
        return f"{amount}塊"
    return str(amount)


def make_sample(item, cat, lo, hi, rectype, rng, allow_account=True, allow_date=True):
    amount = rng.randrange(lo, hi + 1)
    # 金額湊整（贴近真實：很少 137 這種）
    if amount >= 1000:
        amount = int(round(amount, -2))
    elif amount >= 100:
        amount = int(round(amount, -1))
    amount = max(lo, amount)

    amount_txt = _fmt_amount(rng, amount)

    prefix = ""
    ref_offset = 0
    if allow_date and rng.random() < 0.22:
        prefix, ref_offset = rng.choice(DATE_PREFIX)

    suffix = ""
    account = "現金"
    if allow_account and rng.random() < 0.3:
        suffix_txt, account = rng.choice(ACCOUNT_SUFFIX)
        suffix = rng.choice([suffix_txt, " " + suffix_txt, suffix_txt])

    form = rng.randrange(10)
    if form == 0:
        user = f"{prefix}{item}{amount_txt}{suffix}"
    elif form == 1:
        user = f"{prefix}{item} {amount_txt}{suffix}"
    elif form == 2:
        user = f"{prefix}買{item}{amount_txt}{suffix}"
    elif form == 3:
        user = f"{prefix}買{item} {amount_txt}{suffix}"
    elif form == 4 and rectype == "income":
        user = f"{prefix}{item}入帳{amount_txt}"
    elif form == 5 and rectype == "income":
        user = f"{prefix}{item} {amount_txt}入帳"
    elif form == 6 and rng.random() < 0.5:
        user = f"{prefix}{item}{amount_txt}刷卡"
        account = "信用卡"
    elif form in (2, 3) and rectype == "income":
        # 「買」不適用於收入物品 → 改中性動詞
        verb = rng.choice(["領", "收到", "拿到"])
        user = f"{prefix}{verb}{item}{amount_txt}" if form == 2 else f"{prefix}{verb}{item} {amount_txt}"
    else:
        user = f"{prefix}{item}{amount_txt}"

    description = item
    return user, description, amount, account, cat, rectype, ref_offset


def build_row(item, cat, lo, hi, rectype, rng, cat_pool=None, extra_cat_prob=0.25,
              extra_acc_prob=0.2):
    user, desc, amount, account, cat, rectype, ref_offset = make_sample(
        item, cat, lo, hi, rectype, rng)
    ref = date(2026, rng.randrange(1, 13), rng.randrange(1, 29))
    record_date = ref + timedelta(days=ref_offset)

    cats = list(BASE_CATEGORIES)
    if rng.random() < extra_cat_prob:
        extra = rng.sample(EXTRA_CATEGORIES, rng.randrange(1, 3))
        cats = cats + extra
    accs = list(BASE_ACCOUNTS)
    if rng.random() < extra_acc_prob:
        accs = accs + rng.sample(EXTRA_ACCOUNTS, rng.randrange(1, 3))
    if cat not in cats:
        cats.append(cat)
    if account not in accs:
        accs.append(account)

    system = build_system_prompt(["add_record"], ref.isoformat(),
                                 {"category": cats, "account": accs}, weekday=True)
    payload = {"name": "add_record", "args": {
        "amount": amount, "category": cat, "account": account,
        "description": desc, "type": rectype, "date": record_date.isoformat()}}
    assistant = (TOOL_CALL_OPEN + json.dumps(payload, ensure_ascii=False) + TOOL_CALL_CLOSE)
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} -> {path}")


def main():
    rng = random.Random(20261011)
    train_rows, val_rows, test_rows = [], [], []

    # --- test：使用者案例 + 每個物品詞抽一個未見金額/格式（train 不碰這些物品）---
    test_items = [t for t in ALL_ITEMS]
    # 「早餐60」精確案例（fixed amount 60、無前後綴、今天）
    row = build_row("早餐", "飲食", 60, 60, "expense", rng)
    # 強制無前後綴：重新生成並覆寫 user/args
    sysc = row["messages"][0]["content"]
    row["messages"][1]["content"] = "早餐60"
    payload = {"name": "add_record", "args": {
        "amount": 60, "category": "飲食", "account": "現金",
        "description": "早餐", "type": "expense",
        "date": sysc.split("今天是 ")[1][:10]}}
    row["messages"][2]["content"] = (TOOL_CALL_OPEN +
                                     json.dumps(payload, ensure_ascii=False) + TOOL_CALL_CLOSE)
    test_rows.append(row)

    # train 用掉一半物品詞，test 用另一半 → 詞彙零重疊
    # （「早餐」已固定為 test 案例，先從 train 池剔除）
    train_pool = [t for t in ALL_ITEMS if t[0] != "早餐"]
    rng.shuffle(train_pool)
    half = max(24, len(train_pool) // 4)
    test_only = train_pool[:half]
    train_items = train_pool[half:]
    for t in test_only:
        for _ in range(3):  # 每個 test 物品 3 種寫法
            test_rows.append(build_row(*t, rng))
    # test 再加日期/帳戶理解案例
    for t in test_only[:12]:
        test_rows.append(build_row(*t, rng))

    for t in train_items:
        for _ in range(rng.randrange(10, 18)):
            train_rows.append(build_row(*t, rng))
    val_rows = [build_row(*t, rng) for t in train_items for _ in range(2)]

    # train/val 與 test 詞彙零重疊驗證
    def items_of(rows):
        s = set()
        for r in rows:
            a = r["messages"][2]["content"]
            d = json.loads(a[a.index("{"):a.rindex("}") + 1])["args"]["description"]
            s.add(d)
        return s
    leak = items_of(train_rows) & items_of(test_rows)
    assert not leak, f"item leak train->test: {leak}"
    leak2 = items_of(val_rows) & items_of(test_rows)
    assert not leak2, f"item leak val->test: {leak2}"

    dump("dataset/v4/short_train.jsonl", train_rows)
    dump("dataset/v4/short_val.jsonl", val_rows)
    dump("dataset/v4/short_test.jsonl", test_rows)


if __name__ == "__main__":
    main()
