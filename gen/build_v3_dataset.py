"""
gen/build_v3_dataset.py — v3 日期強化資料集組裝器（T6 date boost）

輸入（476b1e4 審核產出 + 守門集）：
    dataset/train_audited.jsonl   2296 筆（LLM 審核通過）
    dataset/test_audited.jsonl     493 筆
    dataset/test_strata.jsonl.bak  831 筆（add_record 永久守門集，原文不動）

輸出（dataset/v3/）：
    train_v3.jsonl          星期注入 + 日期偏移增廣（aug 欄位標記）
    test_v3.jsonl           星期注入（與 runtime prompt 格式對齊）
    gate_v3_weekday.jsonl   守門集星期注入變體（label 零改動，供 v3 模型評測）

組裝規則：
1. 星期注入：前綴「今天是 YYYY-MM-DD。」→「今天是 YYYY-MM-DD（星期X）。」
   星期一律由日期重算（isoweekday()%7，與前端 getDay() 同語意），
   絕不信任任何來源既有星期。注入後做 gate 變體位元驗證：
   移除「（星期X）」後必須與原文逐位元一致。
2. 日期偏移增廣（僅限「純日偏移」相對詞樣本）：
   - 白名單詞：今天/今日/剛剛/昨天/昨晚/前天/明天（語意 = ref 的純日偏移，
     任何位移 Δ 下 (label - ref) 偏移量不變，位移後標註仍正確）
   - 黑名單（拒絕增廣）：週/周/月/號/月底/上個/這個/本月/下 或任何絕對日期
     （X月Y號、YYYY-MM-DD）——這些詞的解析依賴曆法邊界，非位移等變
   - 偏移量：day+2（Δ=+2）、week-3（Δ=-21）；ref 與 label 同步位移，
     偏移量恆等由構造保證；aug 欄位記錄（dataset.py 只讀 messages，多餘鍵無害）
3. Schema 驗證：每筆以 tools.schema.json 靜態檢查
   （name=add_record、required ⊆ keys ⊆ properties、enum ∈ system 池、
   date 格式、type ∈ enum、amount > 0），不合格剔除並報告。
4. 洩漏防呆：train 中凡 user 文本出現在守門集(831)或 test_v3 者一律剔除。

執行：python3 gen/build_v3_dataset.py
"""

import json
import os
import re
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gen.build_prompt import _weekday_zh_for  # noqa: E402
from trainer.validate_json import extract_tool_calls  # noqa: E402

IN_TRAIN = os.path.join(ROOT, "dataset", "train_audited.jsonl")
IN_TEST = os.path.join(ROOT, "dataset", "test_audited.jsonl")
IN_GATE = os.path.join(ROOT, "dataset", "test_strata.jsonl.bak")
OUT_DIR = os.path.join(ROOT, "dataset", "v3")

SCHEMA_PATH = os.path.join(ROOT, "tools.schema.json")

PREFIX_RE = re.compile(r"^今天是 (\d{4}-\d{2}-\d{2})。")
WEEKDAY_INJECT_RE = re.compile(r"^今天是 (\d{4}-\d{2}-\d{2})（星期[日一二三四五六]）。")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ABS_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}\s*月\s*\d{1,2}\s*[號日]")

# 純日偏移白名單（label-ref 偏移不變量在任意 Δ 下成立）
DAY_SHIFT_WORDS = ["今天", "今日", "剛剛", "昨天", "昨晚", "前天", "明天"]
# 曆法邊界依賴詞（拒絕增廣）
RISKY_WORDS = ["週", "周", "月", "號", "月底", "上個", "這個", "本月", "下個", "星期"]

SHIFTS = [("day+2", 2), ("week-3", -21)]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_system(system_content):
    """回傳 (ref_date, category_pool, account_pool)；格式不符回傳 None。"""
    m = re.match(
        r"^今天是 (\d{4}-\d{2}-\d{2})（?(?:星期[日一二三四五六])?）。你是一個記帳助理。"
        r"(?:規則：.*?。)?你被賦予了以下 tools:\n(.*)$",
        system_content,
        re.S,
    )
    if not m:
        return None
    tools = json.loads(m.group(2))
    if isinstance(tools, list):
        tools = tools[0]
    props = tools["parameters"]["properties"]
    return {
        "ref_date": m.group(1),
        "category": props.get("category", {}).get("enum", []),
        "account": props.get("account", {}).get("enum", []),
    }


def inject_weekday(system_content):
    """注入（星期X），星期由 ref_date 重算（不信任任何既有星期）。"""
    m = PREFIX_RE.match(system_content)
    if not m:
        raise ValueError(f"system 前綴格式不符: {system_content[:60]!r}")
    ref = m.group(1)
    return f"今天是 {ref}（{_weekday_zh_for(ref)}）。" + system_content[m.end():]


def strip_weekday(system_content):
    """移除（星期X），供 gate 變體位元驗證。"""
    return WEEKDAY_INJECT_RE.sub(lambda m: f"今天是 {m.group(1)}。", system_content)


def rewrite_ref(system_content, new_ref):
    m = PREFIX_RE.match(system_content)
    if m:
        return f"今天是 {new_ref}（{_weekday_zh_for(new_ref)}）。" + system_content[m.end():]
    m = WEEKDAY_INJECT_RE.match(system_content)
    if m:
        return f"今天是 {new_ref}（{_weekday_zh_for(new_ref)}）。" + system_content[m.end():]
    raise ValueError(f"system 前綴格式不符: {system_content[:60]!r}")


def shift_label_date(label_content, old_date, new_date):
    """把 assistant tool_call 的 date 欄位由 old_date 改為 new_date（位元最小改動）。"""
    old_field = f'"date": "{old_date}"'
    new_field = f'"date": "{new_date}"'
    if label_content.count(old_field) != 1:
        raise ValueError(f"label date 欄位無法唯一定位: {label_content[:120]!r}")
    return label_content.replace(old_field, new_field)


def day_shift_eligible(user_text):
    if ABS_DATE_RE.search(user_text):
        return False
    if any(w in user_text for w in RISKY_WORDS):
        return False
    return any(w in user_text for w in DAY_SHIFT_WORDS)


def static_validate(sample, schema):
    """回傳 issue 清單（空 = 合格）。不用 validate_tool_call 的模糊修正，純靜態。"""
    issues = []
    msgs = sample.get("messages", [])
    if len(msgs) != 3 or [m["role"] for m in msgs] != ["system", "user", "assistant"]:
        issues.append("messages 結構異常")
        return issues
    try:
        parsed_list = extract_tool_calls(msgs[2]["content"])
    except ValueError as e:
        issues.append(f"label 解析失敗: {e}")
        return issues
    if not parsed_list:
        issues.append("label 解析失敗")
        return issues
    if len(parsed_list) != 1:
        issues.append(f"label 含多筆 tool call: {len(parsed_list)}")
        return issues
    parsed = parsed_list[0]
    if parsed["error"]:
        issues.append(f"label 解析失敗: {parsed['error']}")
        return issues
    if parsed["name"] != "add_record":
        issues.append(f"name 非 add_record: {parsed['name']}")
        return issues
    args = parsed["args"]
    if not isinstance(args, dict):
        issues.append("args 非物件")
        return issues
    tool = next(t for t in (schema["tools"] if isinstance(schema, dict) else schema)
                if t["name"] == "add_record")
    props = tool["parameters"]["properties"]
    required = tool["parameters"]["required"]
    pools = parse_system(msgs[0]["content"])
    if pools is None:
        issues.append("system 前綴無法解析")
        return issues
    for k in required:
        if k not in args or args[k] in (None, ""):
            issues.append(f"缺少必要欄位: {k}")
    for k in args:
        if k not in props:
            issues.append(f"未預期欄位: {k}")
    if "amount" in args:
        a = args["amount"]
        if not isinstance(a, (int, float)) or isinstance(a, bool) or a <= 0:
            issues.append(f"amount 無效: {a}")
    if "type" in args and args["type"] not in props["type"]["enum"]:
        issues.append(f"type 非法: {args['type']}")
    if "category" in args and pools["category"] and args["category"] not in pools["category"]:
        issues.append(f"category 不在 system 池: {args['category']}")
    if "account" in args and pools["account"] and args["account"] not in pools["account"]:
        issues.append(f"account 不在 system 池: {args['account']}")
    if "date" in args:
        d = args["date"]
        if not isinstance(d, str) or not DATE_RE.match(d):
            issues.append(f"date 格式非法: {d}")
        else:
            try:
                date.fromisoformat(d)
            except ValueError:
                issues.append(f"date 不存在日曆: {d}")
    return issues


def user_text(sample):
    return sample["messages"][1]["content"]


def main():
    schema = json.load(open(SCHEMA_PATH, encoding="utf-8"))
    train = load_jsonl(IN_TRAIN)
    test = load_jsonl(IN_TEST)
    gate = load_jsonl(IN_GATE)
    print(f"輸入: train_audited={len(train)} test_audited={len(test)} gate831={len(gate)}")

    # ---- 守門集星期注入變體（label 零改動）----
    gate_v3 = []
    for i, s in enumerate(gate):
        sysc = s["messages"][0]["content"]
        injected = inject_weekday(sysc)
        assert strip_weekday(injected) == sysc, f"gate[{i}] 注入非前綴最小改動，中止"
        gate_v3.append({**s, "messages": [dict(s["messages"][0], content=injected),
                                          s["messages"][1], s["messages"][2]]})
    print(f"[OK] gate831 星期注入變體完成（{len(gate_v3)} 筆，移除（星期X）後逐位元回溯驗證通過）")

    # ---- test_v3 星期注入 ----
    test_v3 = []
    for i, s in enumerate(test):
        sysc = s["messages"][0]["content"]
        injected = inject_weekday(sysc)
        assert strip_weekday(injected) == sysc, f"test[{i}] 注入非前綴最小改動，中止"
        test_v3.append({**s, "messages": [dict(s["messages"][0], content=injected),
                                          s["messages"][1], s["messages"][2]]})

    # ---- 洩漏防呆：train 剔除出現在 gate(831) 或 test 的 user 文本 ----
    forbidden = {user_text(s) for s in gate} | {user_text(s) for s in test}
    train_clean, leaked = [], 0
    for s in train:
        if user_text(s) in forbidden:
            leaked += 1
            continue
        train_clean.append(s)
    print(f"[OK] 洩漏防呆: 剔除 {leaked} 筆與守門/test 重疊的 train 樣本（剩 {len(train_clean)}）")

    # ---- train 星期注入 ----
    base_rows = []
    for i, s in enumerate(train_clean):
        sysc = s["messages"][0]["content"]
        injected = inject_weekday(sysc)
        assert strip_weekday(injected) == sysc, f"train[{i}] 注入非前綴最小改動，中止"
        base_rows.append({**s, "messages": [dict(s["messages"][0], content=injected),
                                            s["messages"][1], s["messages"][2]]})

    # ---- 日期偏移增廣（純日偏移白名單樣本）----
    aug_rows = []
    aug_skip = {"not_eligible": 0, "no_label_date": 0, "error": 0}
    for s in base_rows:
        if not day_shift_eligible(user_text(s)):
            aug_skip["not_eligible"] += 1
            continue
        pools = parse_system(s["messages"][0]["content"])
        try:
            parsed = extract_tool_calls(s["messages"][2]["content"])[0]
        except (ValueError, IndexError):
            aug_skip["no_label_date"] += 1
            continue
        label_date = (parsed.get("args") or {}).get("date")
        if not label_date or not DATE_RE.match(label_date):
            aug_skip["no_label_date"] += 1
            continue
        ref = pools["ref_date"]
        try:
            d_ref = date.fromisoformat(ref)
            d_lab = date.fromisoformat(label_date)
        except ValueError:
            aug_skip["no_label_date"] += 1
            continue
        for tag, delta in SHIFTS:
            new_ref = (d_ref + timedelta(days=delta)).isoformat()
            new_lab = (d_lab + timedelta(days=delta)).isoformat()
            try:
                new_sys = rewrite_ref(s["messages"][0]["content"], new_ref)
                new_label = shift_label_date(s["messages"][2]["content"], label_date, new_lab)
            except ValueError:
                aug_skip["error"] += 1
                continue
            aug = {**s,
                   "aug": tag,
                   "messages": [dict(s["messages"][0], content=new_sys),
                                s["messages"][1],
                                dict(s["messages"][2], content=new_label)]}
            # 偏移量恆等斷言（由構造保證，雙重確認）
            nr = parse_system(new_sys)["ref_date"]
            nl = extract_tool_calls(new_label)[0]["args"]["date"]
            assert (date.fromisoformat(nl) - date.fromisoformat(nr)) == \
                   (date.fromisoformat(label_date) - date.fromisoformat(ref)), f"aug {tag} 偏移漂移"
            # 注入後前綴仍必須是「移除（星期X）→ 無星期前綴」的可逆形態
            assert strip_weekday(new_sys) == strip_weekday(
                re.sub(WEEKDAY_INJECT_RE, lambda m: f"今天是 {m.group(1)}。", new_sys)
            )
            aug_rows.append(aug)
    print(f"[OK] 日期偏移增廣: +{len(aug_rows)} 筆 "
          f"(白名單外 {aug_skip['not_eligible']}、無 label date {aug_skip['no_label_date']}、"
          f"錯誤 {aug_skip['error']})")

    # ---- 全量 schema 靜態驗證 ----
    def validate_pass(rows, tag):
        good, bad = [], []
        for s in rows:
            issues = static_validate(s, schema)
            (good if not issues else bad).append((s, issues))
        for s, issues in bad[:5]:
            print(f"  [DROP:{tag}] {issues} :: {s['messages'][1]['content'][:60]!r}")
        print(f"[OK] {tag}: 合格 {len(good)} / 剔除 {len(bad)}")
        return [s for s, _ in good]

    final_train = validate_pass(base_rows + aug_rows, "train_v3")
    final_test = validate_pass(test_v3, "test_v3")
    final_gate = validate_pass(gate_v3, "gate_v3")

    # ---- 寫出 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    dump_jsonl(os.path.join(OUT_DIR, "train_v3.jsonl"), final_train)
    dump_jsonl(os.path.join(OUT_DIR, "test_v3.jsonl"), final_test)
    dump_jsonl(os.path.join(OUT_DIR, "gate_v3_weekday.jsonl"), final_gate)

    n_aug = sum(1 for s in final_train if s.get("aug"))
    print()
    print("=== v3 產出 ===")
    print(f"train_v3.jsonl        {len(final_train)} 筆（基礎 {len(final_train)-n_aug} + 增廣 {n_aug}）")
    print(f"test_v3.jsonl         {len(final_test)} 筆")
    print(f"gate_v3_weekday.jsonl {len(final_gate)} 筆")

    # 難度分佈（train）
    from collections import Counter
    dist = Counter(s.get("difficulty_level", "?") for s in final_train)
    print("train 難度分佈:", dict(dist))

    # 星期正確性全線複核
    bad_wd = 0
    for s in final_train + final_test + final_gate:
        m = WEEKDAY_INJECT_RE.match(s["messages"][0]["content"])
        if not m or _weekday_zh_for(m.group(1)) != re.match(
                r"^今天是 \d{4}-\d{2}-\d{2}（(星期.)）。", s["messages"][0]["content"]).group(1):
            bad_wd += 1
    assert bad_wd == 0, f"星期注入複核失敗 {bad_wd} 筆"
    print("[OK] 全線星期複核: 0 錯位（isoweekday()%7 與前端 getDay 同語意）")


if __name__ == "__main__":
    main()
