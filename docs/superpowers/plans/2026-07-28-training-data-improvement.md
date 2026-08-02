# Training Data Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve category inference (semantic templates) and date generalization (multiple reference dates)

**Architecture:** Modify `batch_generate_v2.py` to use 8 reference dates and B-class semantic templates; add `test_generalization.py` for measurement; retrain with same `train_custom_sft.py`; update `demo.py` for dynamic date

**Tech Stack:** Python, PyTorch, dateutil

---

### Task 1: Add B-class semantic templates and multiple reference dates to batch_generate_v2.py

**Files:**
- Modify: `batch_generate_v2.py`

**Key changes:**
1. `REFERENCE_DATES` → list of 8 dates spanning the year
2. Per-sample: pick random `ref_date`, rebuild maps for that date
3. Add `B_CLASS_EXPENSE_TEXTS` (category-bound semantic templates)
4. Add `B_CLASS_INCOME_TEXTS`
5. Template selection: 60% A-class / 30% B-class / 10% date-focused from colloquial map
6. B-class templates determine category (no random assignment)

- [ ] **Step 1: Change REFERENCE_DATES to multiple dates**

```python
REFERENCE_DATES = [
    date(2026, 1, 5), date(2026, 2, 14),
    date(2026, 5, 15), date(2026, 7, 20),
    date(2026, 8, 3), date(2026, 9, 25),
    date(2026, 11, 8), date(2026, 12, 1),
]
```

- [ ] **Step 2: Add B_CLASS semantic templates**

```python
B_CLASS_EXPENSE_TEXTS = [
    # (category, lambda)
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
]

B_CLASS_INCOME_TEXTS = [
    ("薪資收入", lambda m,a: f"公司發薪水{m}元用{a}"),
    ("薪資收入", lambda m,a: f"月薪{m}元用{a}"),
    ("薪資收入", lambda m,a: f"工資入帳{m}元用{a}"),
    ("獎金紅利", lambda m,a: f"年終獎金{m}元用{a}轉入"),
    ("獎金紅利", lambda m,a: f"分紅{m}元用{a}"),
    ("獎金紅利", lambda m,a: f"績效獎金{m}元用{a}"),
    ("副業外快", lambda m,a: f"接案收入{m}元用{a}"),
    ("副業外快", lambda m,a: f"家教薪水{m}元用{a}"),
    ("副業外快", lambda m,a: f"外包案{m}元用{a}"),
]
```

- [ ] **Step 3: Restructure generate_batch_v2 inner loop for per-sample ref_date**

Change from building maps once at function start to per-sample:

```python
def generate_batch_v2(count_l1=6000, count_l2=4000, count_l3=3000):
    samples = []
    used = set()
    exp_amts, inc_amts = make_amounts(20000, 10000)
    ei = ii = 0
    # ... (next_amt, pick unchanged) ...

    all_templates = L1_EXPENSE_TEXTS + L2_EXPENSE_TEXTS  # A-class
    all_income_templates = L1_INCOME_TEXTS + L2_INCOME_TEXTS

    for _ in range(count_l1 + count_l2 + count_l3):
        ref_date = random.choice(REFERENCE_DATES)
        prefix_map = build_prefix_map(ref_date)
        prefix_list = list(prefix_map.keys())
        colloquial_map = build_colloquial_prefix_map(ref_date)
        colloquial_list = list(colloquial_map.keys())

        rtype = "expense" if random.random() < 0.6 else "income"

        # Template selection: 60% A-class, 30% B-class, 10% date-focused
        tmpl_roll = random.random()
        if tmpl_roll < 0.60:
            # A-class: existing random category
            cat = random.choice(CATEGORIES)
            acc = random.choice(ACCOUNTS)
            amt = next_amt(rtype)
            if not pick("gen", cat, acc, amt, rtype):
                continue
            if rtype == "expense":
                fn = random.choice(all_templates)
                text = fn(cat, acc, amt)
            else:
                fn = random.choice(all_income_templates)
                text = fn(cat, acc, amt)
            diff = "A-class"
        elif tmpl_roll < 0.90:
            # B-class: semantic template, category determined by template
            if rtype == "expense":
                cat, fn = random.choice(B_CLASS_EXPENSE_TEXTS)
            else:
                cat, fn = random.choice(B_CLASS_INCOME_TEXTS)
            acc = random.choice(ACCOUNTS)
            amt = next_amt(rtype)
            if not pick("gen", cat, acc, amt, rtype):
                continue
            text = fn(amt, acc)
            diff = "B-class"
        else:
            # Date-focused: use colloquial map directly
            pref = random.choice(colloquial_list)
            d = colloquial_map[pref]
            cat = random.choice(CATEGORIES)
            acc = random.choice(ACCOUNTS)
            amt = next_amt(rtype)
            if not pick("gen", cat, acc, amt, rtype):
                continue
            text = f"{pref}{cat}消費{amt}元用{acc}"
            diff = "date"
            samples.append(make_sample(diff, cat, acc, amt, rtype, text, d, ref_date))
            continue

        # Date prefix selection
        dr = random.random()
        if diff == "date":
            pass  # already handled above
        elif dr < 0.60:
            pref = random.choice(prefix_list)
            d = prefix_map[pref]
        elif dr < 0.90:
            pref = random.choice(colloquial_list)
            d = colloquial_map[pref]
        else:
            pref = ""
            d = ref_date

        if pref:
            if "今天" in text:
                text = text.replace("今天", pref, 1)
            elif not text.startswith(pref):
                text = pref + text

        samples.append(make_sample(diff, cat, acc, amt, rtype, text, d, ref_date))

    return samples
```

Also update counts in `main()`:
```python
counts = {"A-class": 0, "B-class": 0, "date": 0}
```

- [ ] **Step 4: Verify generation works**

Run: `python batch_generate_v2.py`
Expected: Generates ~15000 samples with varied dates

---

### Task 2: Create test_generalization.py

**Files:**
- Create: `test_generalization.py`

- [ ] **Step 1: Write the complete test file**

```python
import json, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import BookkeepingLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer
from trainer.validate_json import extract_tool_call
from batch_generate_v2 import build_prefix_map, build_colloquial_prefix_map, REFERENCE_DATES
from datetime import date

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BookkeepingTokenizer("jingyaogong/minimind-3")
config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=1024)
model = BookkeepingLM(config)
model.load_state_dict(torch.load("./saves/best_bookkeeping_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

tool_def = {
    "name": "add_record", "description": "新增一筆記帳記錄",
    "parameters": {
        "type": "object", "properties": {
            "amount": {"type": "number"},
            "category": {"type": "string", "enum": ["餐飲飲食", "日常雜貨", "交通出行", "休閒娛樂", "生活繳費"]},
            "account": {"type": "string", "enum": ["現金", "信用卡", "LINE Pay", "悠遊卡"]},
            "description": {"type": "string"},
            "type": {"type": "string", "enum": ["expense", "income"]},
            "date": {"type": "string", "description": "ISO 8601 格式日期，例如 YYYY-MM-DD"}
        }, "required": ["amount", "category", "account", "type", "date"]
    }
}

def generate(expr, ref_date=None):
    if ref_date is None:
        ref_date = date(2026, 7, 20)
    sys_p = f"今天是 {ref_date.isoformat()}。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(tool_def, ensure_ascii=False)}"
    prompt = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{expr}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor(tokenizer.encode(prompt, max_length=1024, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
    curr = input_ids; generated = []
    for _ in range(256):
        with torch.no_grad():
            logits, _ = model(curr)
        next_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        tok = next_id.item()
        if tok == tokenizer.eos_token_id: break
        generated.append(tok)
        curr = torch.cat([curr, next_id], dim=-1)
    out = tokenizer.decode(generated)
    try:
        return extract_tool_call(out).get("date", "NO_DATE")
    except:
        return "PARSE_FAIL"

def test_date_generalization():
    # Test relative dates with DIFFERENT reference dates
    tests = [
        # (expr, ref_date, expected_date)
        ("今天", date(2026, 7, 20), "2026-07-20"),
        ("今天", date(2026, 1, 5), "2026-01-05"),
        ("今天", date(2026, 12, 1), "2026-12-01"),
        ("昨天", date(2026, 7, 20), "2026-07-19"),
        ("昨天", date(2026, 1, 5), "2026-01-04"),
        ("昨天", date(2026, 3, 1), "2026-02-28"),
        ("明天", date(2026, 7, 20), "2026-07-21"),
        ("明天", date(2026, 12, 31), "2027-01-01"),
        ("上週三", date(2026, 7, 20), "2026-07-15"),
        ("下週一", date(2026, 7, 20), "2026-07-27"),
        ("25號", date(2026, 7, 20), "2026-07-25"),
        ("25號", date(2026, 8, 3), "2026-08-25"),
    ]
    passed = 0
    for expr, rd, expected in tests:
        pred = generate(expr, rd)
        ok = "PASS" if pred == expected else "FAIL"
        if ok == "PASS": passed += 1
        print(f"  [{ok}] ref={rd} expr={expr:10s} pred={pred} exp={expected}")
    print(f"  Date generalization: {passed}/{len(tests)} pass")

def test_category_inference():
    # Test natural language → category inference
    tests = [
        ("去餐廳吃飯花了500元用現金", "餐飲飲食"),
        ("搭計程車花了250元用悠遊卡", "交通出行"),
        ("買了一張顯卡15000元用信用卡", "3C電子"),
        ("繳了車險5000元用銀行帳戶", "保險費用"),
        ("看電影300元用信用卡", "休閒娛樂"),
        ("買菜800元用現金", "日常雜貨"),
    ]
    sys_p = f"今天是 2026-07-20。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(tool_def, ensure_ascii=False)}"
    passed = 0
    for expr, expected_cat in tests:
        prompt = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{expr}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor(tokenizer.encode(prompt, max_length=1024, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
        curr = input_ids; generated = []
        for _ in range(256):
            with torch.no_grad():
                logits, _ = model(curr)
            next_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            tok = next_id.item()
            if tok == tokenizer.eos_token_id: break
            generated.append(tok)
            curr = torch.cat([curr, next_id], dim=-1)
        out = tokenizer.decode(generated)
        try:
            args = extract_tool_call(out)
            pred_cat = args.get("category", "")
            ok = "PASS" if pred_cat == expected_cat else "FAIL"
            if ok == "PASS": passed += 1
        except:
            ok = "FAIL"
            pred_cat = "PARSE_FAIL"
        print(f"  [{ok}] expr={expr:30s} pred={pred_cat:10s} exp={expected_cat}")
    print(f"  Category inference: {passed}/{len(tests)} pass")

if __name__ == "__main__":
    print("=== Date Generalization Test ===")
    test_date_generalization()
    print("\n=== Category Inference Test ===")
    test_category_inference()
```

---

### Task 3: Generate new dataset and retrain

- [ ] **Step 1: Generate new dataset**

Run: `python batch_generate_v2.py`
Expected: 15000 samples, validates ok

- [ ] **Step 2: Backup old model**

```bash
copy saves\best_bookkeeping_model.pt saves\best_bookkeeping_model_v1.pt
```

- [ ] **Step 3: Retrain model**

Run: `python train_custom_sft.py --format compressed --epochs 8 --eval_generate --max_eval_samples 3`

Expected: Training runs 8 epochs, best model saved to `saves/best_bookkeeping_model.pt`

---

### Task 4: Run tests and measure improvement

- [ ] **Step 1: Run basic regression test**

Run: `python test_training_format.py`
Expected: 17/17 PASS

- [ ] **Step 2: Run generalization tests**

Run: `python test_generalization.py`
Expected: Measure date + category accuracy

- [ ] **Step 3: Run comprehensive test**

Run: `python test_comprehensive.py`
Expected: ≥ 89% pass rate

---

### Task 5: Update demo.py with dynamic date

- [ ] **Step 1: Change system prompt date to dynamic**

Replace `"今天是 2026-07-20"` with `"今天是 {datetime.now().strftime('%Y-%m-%d')}"` in `FULL_PROMPT`:

```python
from datetime import datetime
# ...
FULL_PROMPT = f"今天是 {datetime.now().strftime('%Y-%m-%d')}。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(TOOL_DEF, ensure_ascii=False)}"
```
