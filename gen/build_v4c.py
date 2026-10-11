"""
gen/build_v4c.py — v4c 短語強化資料產生器

修復 v4b 殘病（根因已查明）：
1. 金額先驗抄錯：「早餐60」→150。根因 NOT few-shot 污染，而是
   (a) 訓練集所有金額都被湊整成十/百的倍數 → 模型學會「金額一定是整數」
   (b) test 物品詞與 train 零重疊 → 「早餐」從未見過，只能瞎猜該品項的典型金額
   解法：物品詞全數入訓練池 + 金額覆蓋極寬（含非整數、破先驗極端值），
         強制學會「從 user 句抄數字」而非「猜典型金額」。
2. category 錯（網費/保險費/家教/借款回收）→ 同因 (a)(b)，入池即解。
3. 泛化探针：另建一批「全新物品詞」test-only，真正測量抄寫機制有無學會。
4. 金額書寫補充：7.69萬 / 1萬2 / 3千5 / 2萬5千 等小數與複合式。

產出：
- dataset/v4c/short_train.jsonl
- dataset/v4c/short_val.jsonl
- dataset/v4c/train_v4c.jsonl  (train_v3 + short_train)
- dataset/v4c/val_v4c.jsonl     (test_v3 + short_val)
- dataset/v4c/probe_unseen.jsonl  全新物品詞 test-only（泛化探针）
- dataset/v4c/regression_v4.jsonl 原 short_test（生產回歸，含「早餐60」）
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
from build_short_phrase import (
    EXPENSE_ITEMS, INCOME_ITEMS, ALL_ITEMS, BASE_CATEGORIES, EXTRA_CATEGORIES,
    BASE_ACCOUNTS, EXTRA_ACCOUNTS, ACCOUNT_SUFFIX, DATE_PREFIX,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
)

# 全新物品詞（不出現在任何訓練池）→ 泛化探针專用
UNSEEN_ITEMS = [
    ("按摩", "娛樂", 300, 1500, "expense"),
    ("寵物健檢", "醫療", 800, 4000, "expense"),
    ("貓飼料", "醫療", 300, 1800, "expense"),
    ("狗飼料", "醫療", 400, 2000, "expense"),
    ("停車月票", "交通", 900, 2500, "expense"),
    ("機車稅金", "其他", 1200, 4800, "expense"),
    ("垃圾袋費", "其他", 30, 200, "expense"),
    ("社區清潔費", "其他", 300, 1500, "expense"),
    ("鋼琴課", "教育", 800, 3000, "expense"),
    ("游泳課", "教育", 400, 2000, "expense"),
    ("健身月繳", "娛樂", 500, 2200, "expense"),
    ("網球場地費", "娛樂", 200, 900, "expense"),
    ("相機維修", "日常", 400, 3500, "expense"),
    ("鎖匠", "日常", 300, 1800, "expense"),
    ("除蟲", "日常", 600, 3000, "expense"),
    ("兼職加班費", "兼職", 500, 5000, "income"),
    ("稿費", "兼職", 800, 12000, "income"),
    ("翻譯費", "兼職", 600, 9000, "income"),
    ("房東退押金", "欠款回收", 3000, 22000, "income"),
    ("合夥分紅", "獎金", 2000, 30000, "income"),
]


def _fmt_amount_c(rng, amount):
    """渲染金額書寫，覆蓋整數/小數萬/複合千/非整數。"""
    style = rng.random()
    if amount >= 10000:
        if style < 0.18:
            return f"{amount / 10000:.2f}".rstrip("0").rstrip(".") + "萬"
        if style < 0.32 and amount % 1000 == 0:
            w, t = divmod(amount, 10000)
            return f"{w}萬{t // 1000}" if t else f"{w}萬"
        if style < 0.42 and amount % 1000 == 0:
            w, t = divmod(amount, 10000)
            return f"{w}萬{t // 1000}千" if t else f"{w}萬"
        if style < 0.52:
            return f"{amount // 1000}k"
    elif amount >= 1000:
        if style < 0.25:
            q, r = divmod(amount, 1000)
            return f"{q}千{r // 100}" if r and r % 100 == 0 else f"{q}千"
        if style < 0.35 and amount % 100 == 0:
            return f"{amount // 1000}千{amount % 1000 // 100}"
    if style < 0.12:
        return f"{amount}元"
    if style < 0.2:
        return f"{amount}塊"
    return str(amount)


def make_sample_c(item, cat, lo, hi, rectype, rng):
    """取樣規則：40% 典型區間、35% 全區間、25% 破先驗極端值；35% 不湊整。"""
    roll = rng.random()
    if roll < 0.40:
        amount = rng.randrange(lo, hi + 1)
    elif roll < 0.75:
        amount = rng.randrange(max(10, int(lo * 0.5)), int(hi * 1.6) + 1)
    else:
        # 破先驗：故意取該品項「極不可能」的金額（強化管理抄寫）
        if rng.random() < 0.5:
            amount = rng.randrange(max(10, int(lo * 0.15)), max(20, int(lo * 0.6)))
        else:
            amount = rng.randrange(int(hi * 1.8), int(hi * 4.5) + 1)

    if rng.random() < 0.65:  # 65% 湊整（貼近真實），35% 保留非整數
        if amount >= 1000:
            amount = int(round(amount, -2))
        elif amount >= 100:
            amount = int(round(amount, -1))
    amount = max(1, amount)

    amount_txt = _fmt_amount_c(rng, amount)

    prefix = ""
    ref_offset = 0
    if rng.random() < 0.30:
        prefix, ref_offset = rng.choice(DATE_PREFIX)

    suffix = ""
    account = "現金"
    if rng.random() < 0.35:
        s_txt, account = rng.choice(ACCOUNT_SUFFIX)
        suffix = rng.choice([s_txt, " " + s_txt])

    form = rng.randrange(12)
    if form <= 3:
        user = f"{prefix}{item}{' ' if form % 2 else ''}{amount_txt}{suffix}"
    elif form == 4 and rectype == "income":
        user = f"{prefix}{item}入帳{amount_txt}"
    elif form == 5 and rectype == "income":
        user = f"{prefix}{item} {amount_txt}入帳"
    elif form == 6 and rectype == "income":
        verb = rng.choice(["領", "收到", "拿到"])
        user = f"{prefix}{verb}{item}{amount_txt}"
    elif form == 7 and rng.random() < 0.5:
        user = f"{prefix}{item}{amount_txt}刷卡"
        account = "信用卡"
    elif form == 8 and rectype == "expense":
        user = f"{prefix}買{item}{amount_txt}{suffix}"
    elif form == 9 and rectype == "expense":
        user = f"{prefix}買{item} {amount_txt}{suffix}"
    else:
        user = f"{prefix}{item}{amount_txt}"

    return user, item, amount, account, cat, rectype, ref_offset


def build_row_c(item, cat, lo, hi, rectype, rng, extra_cat_prob=0.3, extra_acc_prob=0.22):
    user, desc, amount, account, cat, rectype, ref_offset = make_sample_c(
        item, cat, lo, hi, rectype, rng)
    ref = date(2026, rng.randrange(1, 13), rng.randrange(1, 29))
    record_date = ref + timedelta(days=ref_offset)

    cats = list(BASE_CATEGORIES)
    if rng.random() < extra_cat_prob:
        cats = cats + rng.sample(EXTRA_CATEGORIES, rng.randrange(1, 3))
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
    assistant = TOOL_CALL_OPEN + json.dumps(payload, ensure_ascii=False) + TOOL_CALL_CLOSE
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
    rng = random.Random(20261012)

    train_rows = []
    for t in ALL_ITEMS:  # 全數入池（含 早餐/網費/保險費/家教/借款回收）
        args = list(t)
        for _ in range(rng.randrange(26, 34)):
            train_rows.append(build_row_c(args[0], args[1], args[2], args[3], args[4], rng))
    val_rows = []
    for t in ALL_ITEMS:
        args = list(t)
        for _ in range(3):
            val_rows.append(build_row_c(args[0], args[1], args[2], args[3], args[4], rng))

    # 泛化探针：全新物品詞，train 絕未見過
    probe_rows = []
    for t in UNSEEN_ITEMS:
        args = list(t)
        for _ in range(8):
            probe_rows.append(build_row_c(args[0], args[1], args[2], args[3], args[4], rng))

    # train/val 與 probe 物品詞零重疊驗證
    def items_of(rows):
        s = set()
        for r in rows:
            a = r["messages"][2]["content"]
            s.add(json.loads(a[a.index("{"):a.rindex("}") + 1])["args"]["description"])
        return s
    leak = (items_of(train_rows) | items_of(val_rows)) & items_of(probe_rows)
    assert not leak, f"item leak -> probe: {leak}"

    dump("dataset/v4c/short_train.jsonl", train_rows)
    dump("dataset/v4c/short_val.jsonl", val_rows)
    dump("dataset/v4c/probe_unseen.jsonl", probe_rows)

    # 合併 v3
    v3_train = [l.rstrip("\n") for l in open("dataset/v3/train_v3.jsonl", encoding="utf-8") if l.strip()]
    v3_val = [l.rstrip("\n") for l in open("dataset/v3/test_v3.jsonl", encoding="utf-8") if l.strip()]
    with open("dataset/v4c/train_v4c.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(v3_train) + "\n")
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("dataset/v4c/val_v4c.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(v3_val) + "\n")
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_v4c = v3 {len(v3_train)} + short {len(train_rows)} = {len(v3_train)+len(train_rows)}")
    print(f"val_v4c   = v3 {len(v3_val)} + short {len(val_rows)} = {len(v3_val)+len(val_rows)}")

    # 生產回歸集 = 原 short_test（含「早餐60」）
    src = [l.rstrip("\n") for l in open("dataset/v4/short_test.jsonl", encoding="utf-8") if l.strip()]
    with open("dataset/v4c/regression_v4.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(src) + "\n")
    print(f"regression_v4 = {len(src)} (原 short_test，含 早餐60)")


if __name__ == "__main__":
    main()
