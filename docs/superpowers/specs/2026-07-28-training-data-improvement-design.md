# Training Data Improvement: Semantic Categories & Variable Reference Dates

## Goal

Fix two root causes of poor model generalization:
1. **Category misclassification**: Model memorizes random category→text mappings instead of inferring from semantics
2. **Date always 2026-07-13**: Model memorizes absolute date mappings from a single reference date (2026-07-20)

## Approach

### 1. Multiple Reference Dates

Replace single `REFERENCE_DATES = [date(2026, 7, 20)]` with diverse dates:

```python
REFERENCE_DATES = [
    date(2026, 1, 5), date(2026, 2, 14),  # Winter/Spring
    date(2026, 5, 15), date(2026, 7, 20),  # Summer (keep original)
    date(2026, 8, 3), date(2026, 9, 25),   # Late Summer/Fall
    date(2026, 11, 8), date(2026, 12, 1),  # Winter
]
```

Per-sample generation flow:
1. Pick random `ref_date` from `REFERENCE_DATES`
2. Build `prefix_map` and `colloquial_map` for that `ref_date`
3. System prompt becomes `"今天是 {ref_date.isoformat()}。你是一個記帳助理。..."`
4. Training dates are computed relative to this `ref_date`
5. If `ref_date.day` is too large (e.g., 31 doesn't exist in Feb), clamp to month length

### 2. Semantic Categories

Keep existing "explicit" templates (A-class, ~60%) where category name appears in text verbatim.
Add new "inference" templates (B-class, ~30%) where category must be inferred from context.

B-class templates by category:

```
餐飲飲食: ["去餐廳吃飯花了{m}元用{a}", "買了杯咖啡{m}元用{a}", "叫了外送{m}元用{a}", "朋友聚餐{m}元用{a}"]
交通出行: ["搭計程車花了{m}元用{a}", "加油{m}元用{a}", "坐捷運{m}元用{a}", "停車費{m}元用{a}"]
3C電子:   ["買了一張顯卡{m}元用{a}", "買了一條充電線{m}元用{a}", "換了新手機{m}元用{a}", "買了滑鼠{m}元用{a}"]
保險費用: ["繳了車險{m}元用{a}", "付了醫療險{m}元用{a}", "繳了保險費{m}元用{a}"]
日常雜貨: ["買菜{m}元用{a}", "超市購物{m}元用{a}", "買了牛奶麵包{m}元用{a}"]
休閒娛樂: ["看電影{m}元用{a}", "買了遊戲{m}元用{a}", "唱歌{m}元用{a}"]
醫療保健: ["看醫生掛號費{m}元用{a}", "買藥{m}元用{a}", "牙醫{m}元用{a}"]
教育學習: ["買書{m}元用{a}", "繳學費{m}元用{a}", "報名課程{m}元用{a}"]
寵物支出: ["買飼料{m}元用{a}", "寵物看醫生{m}元用{a}", "買貓砂{m}元用{a}"]
服飾美妝: ["買衣服{m}元用{a}", "買化妝品{m}元用{a}", "買鞋子{m}元用{a}"]
獎金紅利: ["年終獎金{m}元用{a}轉入", "分紅{m}元用{a}", "績效獎金{m}元用{a}"]
副業外快: ["接案收入{m}元用{a}", "家教薪水{m}元用{a}", "外包案{m}元用{a}"]
薪資收入: ["公司發薪水{m}元用{a}", "月薪{m}元用{a}", "工資入帳{m}元用{a}"]
```

### 3. Test Suite for Measuring Improvement

New test file: `test_generalization.py`

**Date tests:**
- Same query with different reference dates → should produce correct relative dates
- Day-only queries like "25號" with reference date in different months
- Relative dates like "上週三" with different reference dates
- Year/month crossing boundaries

**Category tests:**
- Natural language queries without explicit category keyword
- Verify correct category inference from context

**Regression test:**
- The existing 17 basic cases must still pass
- The existing comprehensive test categories should not regress

### 4. Data Generation

Modify `batch_generate_v2.py`:
- Accept `REFERENCE_DATE` parameter per sample
- `build_prefix_map(date)` already works with any date
- `build_colloquial_prefix_map(date)` already works with any date
- System prompt date comes from the sampled reference date

### 5. Retraining

Use `train_custom_sft.py` same parameters (8 epochs, lr=2e-4, batch=16).

### 6. Demo

System prompt date changes from hardcoded to dynamic: `datetime.now()`.
Keeping the full TOOL_DEF to maintain format compatibility.

## Success Criteria

- Date generalization test: ≥ 90% pass rate across varied reference dates
- Category test: ≥ 80% pass rate for natural language inference
- Basic regression: 17/17 basic tests still pass
- Overall comprehensive test: ≥ 90% pass rate
