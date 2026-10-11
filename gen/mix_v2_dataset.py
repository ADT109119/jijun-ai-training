"""
gen/mix_v2_dataset.py — P1 混合資料集：舊 add_record (4-tool 化) + 新 task 資料

流程：
1. 載入舊 train/test .bak（原始格式：單工具 prompt + 原始日期）
2. 以 convert_dataset_dates.convert_sample 還原 07-20 日期錨定（與 v1 訓練時一致）
3. 驗證：test .bak 重新轉換後必須與 test_strata.jsonl.bak_prefix 逐筆一致
   （驗證我们对 v1 訓練資料管線的逆向理解正確）
4. 重寫舊資料 system prompt → Protocol v2 4-tool 格式（enum 保留各樣本子集，
   新增 contact 子集；add_record tool def 位元不變，僅包進陣列）
5. 載入新 v2 raw（dataset/v2_*/raw.jsonl），以各樣本自己的 system prompt enum
   重新驗證 label（防生成器 bug 的最後一道門）
6. 分層 80/20 切分（新資料；舊資料沿用既有 train/test 分界）
7. 輸出 dataset/v2/train_v2.jsonl + test_v2.jsonl + 統計報告

用法：
    python gen/mix_v2_dataset.py [--dry-run]
"""

import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

from convert_dataset_dates import convert_sample
from gen.build_prompt import build_system_prompt

V2_TOOL_SET = ["add_record", "transfer", "add_debt", "settle_debt"]
OLD_TRAIN = os.path.join(ROOT, "dataset", "train_strata.jsonl.bak")
OLD_TEST = os.path.join(ROOT, "dataset", "test_strata.jsonl.bak")
OLD_TEST_PREFIX = os.path.join(ROOT, "dataset", "test_strata.jsonl.bak_prefix")
NEW_DIRS = {
    "transfer": os.path.join(ROOT, "dataset", "v2_transfer", "raw.jsonl"),
    "add_debt": os.path.join(ROOT, "dataset", "v2_add_debt", "raw.jsonl"),
    "settle_debt": os.path.join(ROOT, "dataset", "v2_settle_debt", "raw.jsonl"),
    "add_record": os.path.join(ROOT, "dataset", "v2_add_record", "raw.jsonl"),  # date_boost 批次
}
OUT_DIR = os.path.join(ROOT, "dataset", "v2")


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # 生成進行中可能留下半行；跳過並警告
                print(f"  [warn] {path}:{ln} 非完整 JSON，跳過（可能是生成中的半行）")
    return out


def parse_v1_tool_def(system_content):
    """從舊 v1 system prompt 解析嵌入的 tool def（取 enum 用）。"""
    marker = "你被賦予了以下 tools:\n"
    idx = system_content.find(marker)
    if idx < 0:
        return None
    rest = system_content[idx + len(marker):].strip()
    try:
        return json.loads(rest)
    except json.JSONDecodeError:
        return None


def rewrite_old_sample(sample, idx, rng):
    """舊 add_record 樣本 → v2 4-tool 格式。回傳 (new_sample, ok, reason)。"""
    msgs = sample.get("messages", [])
    sys_msg = next((m for m in msgs if m["role"] == "system"), None)
    if not sys_msg:
        return None, False, "missing system"
    tool_def = parse_v1_tool_def(sys_msg["content"])
    if not tool_def:
        return None, False, "cannot parse v1 tool def"

    m = re.search(r"^今天是\s*(\d{4}-\d{2}-\d{2})", sys_msg["content"])
    anchor = m.group(1) if m else "2026-07-20"
    from datetime import date as _date
    y, mo, d = map(int, anchor.split("-"))
    ref = _date(y, mo, d)

    cats = tool_def.get("parameters", {}).get("properties", {}).get("category", {}).get("enum", [])
    accts = tool_def.get("parameters", {}).get("properties", {}).get("account", {}).get("enum", [])
    if not cats or not accts:
        return None, False, "missing enum in v1 tool def"

    # contact 子集（rng 由 sample idx 種子，可重現）
    from generate_dataset import CONTACTS
    contacts = rng.sample(CONTACTS, k=5)

    new_system = build_system_prompt(
        V2_TOOL_SET, ref,
        {"category": cats, "account": accts, "contact": contacts},
        weekday=False, compact=False,
    )

    new_sample = json.loads(json.dumps(sample))  # deep copy
    for msg in new_sample["messages"]:
        if msg["role"] == "system":
            msg["content"] = new_system
    new_sample["task"] = "add_record"
    new_sample["v2_rewritten"] = True
    return new_sample, True, "ok"


def verify_test_conversion():
    """重放 convert_sample 於 test .bak，與 .bak_prefix 逐筆比對。"""
    originals = load_jsonl(OLD_TEST)
    expected = load_jsonl(OLD_TEST_PREFIX)
    if len(originals) != len(expected):
        print(f"[FAIL] 筆數不符: originals={len(originals)}, prefix={len(expected)}")
        return False
    mismatches = 0
    for i, (o, e) in enumerate(zip(originals, expected)):
        converted = convert_sample(json.loads(json.dumps(o)), i)
        if json.dumps(converted, ensure_ascii=False, sort_keys=True) != json.dumps(e, ensure_ascii=False, sort_keys=True):
            mismatches += 1
            if mismatches <= 3:
                print(f"  [mismatch] sample {i}:")
                print(f"    converted: {json.dumps(converted, ensure_ascii=False)[:200]}")
                print(f"    expected : {json.dumps(e, ensure_ascii=False)[:200]}")
    if mismatches:
        print(f"[FAIL] {mismatches}/{len(originals)} 筆轉換結果與 .bak_prefix 不符")
        return False
    print(f"[OK] test .bak → convert_sample 與 .bak_prefix 逐筆一致（{len(originals)} 筆）")
    return True


def validate_new_sample(sample):
    """以樣本自己的 system prompt enum 驗證 label。回傳 (ok, reason)。"""
    from trainer.validate_json import extract_tool_calls, validate_tool_call
    from gen.build_prompt import load_schema

    msgs = sample.get("messages", [])
    sys_c = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user_c = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_c = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    if not all([sys_c, user_c, asst_c]):
        return False, "missing role content"

    # 解析 sample 自己的 enum 池
    marker = "你被賦予了以下 tools:\n"
    idx = sys_c.find(marker)
    if idx < 0:
        return False, "no tools marker in system"
    try:
        tools = json.loads(sys_c[idx + len(marker):].strip())
    except json.JSONDecodeError as e:
        return False, f"tools json parse fail: {e}"
    pools = {"category": [], "account": [], "contact": []}
    tmap = {t["name"]: t for t in tools}
    for prop_name, pool_key in (("category", "category"), ("account", "account"), ("contact", "contact")):
        for t in tools:
            props = t.get("parameters", {}).get("properties", {})
            if prop_name in props and "enum" in props[prop_name]:
                pools[pool_key] = props[prop_name]["enum"]
                break

    try:
        calls = extract_tool_calls(asst_c)
    except ValueError as e:
        return False, f"extract fail: {e}"
    if len(calls) != 1:
        return False, f"expected 1 call, got {len(calls)}"
    call = calls[0]
    if call["error"]:
        return False, call["error"]

    declared_task = sample.get("task")
    if call["name"] != declared_task:
        return False, f"task mismatch: declared={declared_task}, actual={call['name']}"

    vres = validate_tool_call(call["name"], call["args"], pools)
    if not vres["valid"] or vres["needs_confirmation"]:
        return False, f"validation: {vres['issues']}"
    return True, "ok"


def stratified_split(samples, ratio=0.8, seed=42):
    """依 (task, difficulty) 分層的 80/20 切分。"""
    rng = random.Random(seed)
    groups = {}
    for s in samples:
        key = (s.get("task", "unknown"), s.get("difficulty_level", "unknown"))
        groups.setdefault(key, []).append(s)
    train, test = [], []
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        n_train = int(len(items) * ratio)
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    dry_run = "--dry-run" in sys.argv
    print("=== P1 混合資料集建置 ===\n")

    # 1-3. 驗證轉換管線
    if not verify_test_conversion():
        print("\n[ABORT] 舊資料轉換驗證失敗，中斷。")
        sys.exit(1)

    # 4. 舊資料 4-tool 化
    print("\n--- 舊 add_record 資料 4-tool 化 ---")
    rng = random.Random(20260720)
    old_train = load_jsonl(OLD_TRAIN)
    old_test_prefix = load_jsonl(OLD_TEST_PREFIX)  # 已驗證 = convert(.bak)，v1 訓練實際格式
    # 舊 train：.bak 是原始檔（轉換前的備份），先還原 07-20 日期錨定再重寫 system
    rewritten_train, rewritten_test = [], []
    for i, s in enumerate(old_train):
        converted = convert_sample(json.loads(json.dumps(s)), i)
        new, ok, reason = rewrite_old_sample(converted, i, rng)
        if ok:
            rewritten_train.append(new)
        else:
            print(f"  [skip train {i}] {reason}")
    for i, s in enumerate(old_test_prefix):
        new, ok, reason = rewrite_old_sample(s, 100000 + i, rng)
        if ok:
            rewritten_test.append(new)
        else:
            print(f"  [skip test {i}] {reason}")
    print(f"  舊 train: {len(rewritten_train)}/{len(old_train)} 筆重寫成功")
    print(f"  舊 test : {len(rewritten_test)}/{len(old_test_prefix)} 筆重寫成功")

    # 5. 新資料載入 + 驗證
    print("\n--- 新 v2 資料驗證 ---")
    new_all = []
    new_stats = {}
    for task, path in NEW_DIRS.items():
        if not os.path.exists(path):
            print(f"  [warn] {task}: 找不到 {path}（尚未生成？）")
            continue
        samples = load_jsonl(path)
        ok_n = 0
        bad_reasons = {}
        for s in samples:
            ok, reason = validate_new_sample(s)
            if ok:
                ok_n += 1
                new_all.append(s)
            else:
                bad_reasons[reason[:40]] = bad_reasons.get(reason[:40], 0) + 1
        new_stats[task] = (len(samples), ok_n, bad_reasons)
        print(f"  {task}: {ok_n}/{len(samples)} 通過驗證" + (f"（淘汰原因: {bad_reasons}）" if bad_reasons else ""))

    # 6. 切分：舊資料沿用既有 train/test 分界；新資料分層 80/20
    new_train, new_test = stratified_split(new_all)

    train_v2 = rewritten_train + new_train
    test_v2 = rewritten_test + new_test
    rng2 = random.Random(42)
    rng2.shuffle(train_v2)
    rng2.shuffle(test_v2)

    # 7. 報告 + 輸出
    def report(name, data):
        by_task = {}
        by_diff = {}
        date_boost = 0
        for s in data:
            by_task[s.get("task", "?")] = by_task.get(s.get("task", "?"), 0) + 1
            d = s.get("difficulty_level", "?")
            by_diff[d] = by_diff.get(d, 0) + 1
            if s.get("date_boost"):
                date_boost += 1
        print(f"\n=== {name} ({len(data)} 筆) ===")
        print(f"  by task: {dict(sorted(by_task.items()))}")
        print(f"  by difficulty: {dict(sorted(by_diff.items()))}")
        print(f"  date_boost 樣本: {date_boost}")
        if by_task.get("add_record"):
            print(f"  add_record 佔比: {by_task['add_record'] / len(data) * 100:.1f}%")

    report("TRAIN v2", train_v2)
    report("TEST v2", test_v2)

    if dry_run:
        print("\n[dry-run] 不寫檔。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "train_v2.jsonl")
    test_path = os.path.join(OUT_DIR, "test_v2.jsonl")
    for path, data in ((train_path, train_v2), (test_path, test_v2)):
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n已寫入 {path} ({len(data)} 筆)")


if __name__ == "__main__":
    main()
