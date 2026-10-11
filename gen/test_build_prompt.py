"""
gen/test_build_prompt.py — P0 位元級相容性驗證（黃金測試）

核心主張：gen/build_prompt.py 用 tools.schema.json 渲染出的 add_record prompt，
必須與現有訓練資料（dataset/raw_filtered.jsonl）的 system content 逐位元一致。
若此測試通過，證明「單一真相源」可以無損接管現有 prompt 生成，v1 模型不受影響。

執行：python3 gen/test_build_prompt.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen.build_prompt import build_system_prompt, render_tool_def, load_schema  # noqa: E402

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "dataset", "raw_filtered.jsonl")

WEEKDAY_RE = re.compile(r"^今天是 (\d{4}-\d{2}-\d{2})（星期[日一二三四五六]）")


def extract_enums_from_system(system_content):
    """從真實訓練資料的 system content 還原 (ref_date, category_pool, account_pool)。"""
    m = re.match(r"^今天是 (\d{4}-\d{2}-\d{2})。你是一個記帳助理。你被賦予了以下 tools:\n(.*)$",
                 system_content, re.S)
    assert m, f"system content 不符合 v1 前綴格式: {system_content[:60]!r}"
    ref_date, tools_json = m.group(1), m.group(2)
    tool = json.loads(tools_json)
    props = tool["parameters"]["properties"]
    return ref_date, props["category"]["enum"], props["account"]["enum"]


def main():
    schema = load_schema()
    total = 0
    mismatches = []
    multi_tool_checked = False

    with open(DATASET, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if total >= 400:
                break
            d = json.loads(line)
            total += 1
            system = next(m["content"] for m in d["messages"] if m["role"] == "system")
            ref_date, cats, accts = extract_enums_from_system(system)

            # 1) 位元級再現（spaced JSON、無星期 — v1 訓練格式）
            rebuilt = build_system_prompt(["add_record"], ref_date,
                                          {"category": cats, "account": accts},
                                          schema=schema)
            if rebuilt != system:
                mismatches.append(idx)
                if len(mismatches) <= 3:
                    print(f"[FAIL] line {idx}\n  原始: {system[:200]!r}\n  重建: {rebuilt[:200]!r}")

            # 2) compact 變體 = 前端 JSON.stringify 格式（與 v1 生產 prompt 對齊驗證）
            compact = build_system_prompt(["add_record"], ref_date,
                                          {"category": cats, "account": accts},
                                          compact=True, schema=schema)
            expected_compact = system.replace("。你是一個記帳助理", "。你是一個記帳助理", 1)
            # 前端格式差異僅在 JSON 空位（冒號/逗號後空格）
            tools_obj = json.loads(system.split("\n", 1)[1])
            expected_compact = system.split("\n", 1)[0] + "\n" + json.dumps(
                tools_obj, ensure_ascii=False, separators=(",", ":"))
            assert compact == expected_compact, f"line {idx}: compact 變體不等"

            # 3) 抽樣驗證 weekday 模式與多工具模式（不要求位元、只要求結構合法）
            if idx == 0:
                with_wd = build_system_prompt(["add_record"], ref_date,
                                              {"category": cats, "account": accts},
                                              weekday=True, schema=schema)
                assert WEEKDAY_RE.match(with_wd), f"weekday 模式格式錯誤: {with_wd[:40]!r}"
                assert "。你是一個記帳助理" in with_wd

            if idx == 1 and not multi_tool_checked:
                contacts = ["小明", "阿寶"]
                multi = build_system_prompt(
                    ["add_record", "transfer", "add_debt", "settle_debt"], ref_date,
                    {"category": cats, "account": accts, "contact": contacts},
                    schema=schema)
                arr = json.loads(multi.split("\n", 1)[1])
                assert isinstance(arr, list) and len(arr) == 4, "多工具應為 4 元素 JSON 陣列"
                assert [t["name"] for t in arr] == ["add_record", "transfer", "add_debt", "settle_debt"]
                # 確認無內部欄位洩漏
                blob = json.dumps(arr, ensure_ascii=False)
                assert "enum_source" not in blob and "validation" not in blob
                multi_tool_checked = True
                print("[OK] 多工具格式：JSON 陣列、順序保留、無 enum_source/validation 洩漏")

    assert multi_tool_checked, "未執行到多工具驗證"
    if mismatches:
        print(f"\n[FAIL] {len(mismatches)}/{total} 筆 system prompt 位元不一致: {mismatches[:10]}")
        sys.exit(1)
    print(f"\n[PASS] {total} 筆真實訓練資料全部位元級再現（spaced v1 格式 + compact 前端變體）")


def test_weekday_goldens():
    """星期映射金標：前端用 getDay()（Sunday=0），Python 必須用 isoweekday()%7。

    舊版 _weekday_zh_for 誤用 date.weekday()%7（Monday=0），造成整體錯位一天
    （星期日→星期六、星期一→星期日）。此測試鎖死正確映射，防止回歸。
    """
    from datetime import date as _date

    from gen.build_prompt import _weekday_zh_for

    goldens = {
        "2026-10-10": "星期六",   # isoweekday()=6
        "2026-10-11": "星期日",   # isoweekday()=7 → index 0
        "2026-10-12": "星期一",   # isoweekday()=1
        "2026-01-01": "星期四",
        "2024-02-29": "星期四",  # 閏年邊界（2024-02-29 存在）
        "2026-12-31": "星期四",
    }
    for dstr, expect in goldens.items():
        got = _weekday_zh_for(dstr)
        assert got == expect, f"{dstr}: 期望 {expect} 得到 {got}"
        # 與 JS getDay() 語意逐日交叉驗證（連續 14 天）
        y, m, d = map(int, dstr.split("-"))
        js_idx = _date(y, m, d).isoweekday() % 7
        assert _WEEKDAY_LIST_JS[js_idx] == got, f"{dstr}: 與前端 getDay 語意不符"

    # 連續 30 天全線掃描：不得出現錯位（星期X 必須與 isoweekday 對齊）
    import datetime as _dt
    for i in range(30):
        cur = _dt.date(2026, 3, 1) + _dt.timedelta(days=i)
        dstr = cur.isoformat()
        assert _weekday_zh_for(dstr) == _WEEKDAY_LIST_JS[cur.isoweekday() % 7], dstr

    print("[OK] 星期映射金標 + 連續 30 天掃描通過（與前端 getDay 語意一致）")


_WEEKDAY_LIST_JS = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]


if __name__ == "__main__":
    test_weekday_goldens()
    main()
