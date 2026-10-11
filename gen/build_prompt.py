"""
gen/build_prompt.py — Protocol v2 system prompt 生成器（P0 地基）

單一真相源：../tools.schema.json
本模組負責把 schema + 動態 enum 池 渲染成與現有訓練資料「位元一致」的 system prompt。

用法：
    from gen.build_prompt import build_system_prompt, render_tool_def

    # v1 相容（無星期、spaced JSON）— 與 generate_dataset.py 現有輸出位元一致
    prompt = build_system_prompt(["add_record"], "2026-01-26",
                                 {"category": cats, "account": accts})

    # 多工具（P1+）— tools 部分為 JSON 陣列
    prompt = build_system_prompt(["add_record", "transfer", "add_debt"],
                                 "2026-03-01",
                                 {"category": cats, "account": accts, "contact": contacts},
                                 weekday=True)

位元規則（與現有 1672 筆訓練資料對齊）：
- 前綴：今天是 {ref_date}。你是一個記帳助理。你被賦予了以下 tools:\n
  （v1 訓練資料無星期；weekday=True 時插入（星期X），供 v2 批次使用）
- tools_json：
  - 單 tool → 單一物件 {...}（v1 格式，也是所有 v1 訓練資料的格式）
  - 多 tool → JSON 陣列 [{...}, {...}]（v2 新格式，P1 模型起使用）
- 序列化：ensure_ascii=False + 預設 separators (", ", ": ")，等同 Python json.dumps
- enum_source 欄位在渲染時替換為實際 enum 值，且不出現於輸出
- validation 欄位只供 validator 使用，不出現在 prompt
"""

import json
import os

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "tools.schema.json")
_schema_cache = None


def load_schema(path=None):
    global _schema_cache
    if path is None and _schema_cache is not None:
        return _schema_cache
    with open(path or _SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    if path is None:
        _schema_cache = schema
    return schema


def _get_tool(schema, name):
    for tool in schema["tools"]:
        if tool["name"] == name:
            return tool
    raise KeyError(f"未知 tool: {name}（schema 定義: {[t['name'] for t in schema['tools']]}）")


def render_tool_def(tool, enums):
    """把一個 tool 定義渲染成 prompt 用的 JSON 結構（dict）。

    - enum_source → 替換為 enums[source] 清單
    - 保持 properties 原始順序（位元穩定性）
    - 剔除 enum_source / 其他非 prompt 欄位
    """
    properties = {}
    for prop_name, prop in tool["parameters"]["properties"].items():
        rendered = {}
        for k, v in prop.items():
            if k == "enum_source":
                pool = enums.get(v)
                if pool is None:
                    raise ValueError(
                        f"tool {tool['name']} 欄位 {prop_name} 需要動態 enum 池 '{v}'，但未提供"
                    )
                rendered["enum"] = list(pool)
            else:
                rendered[k] = v
        properties[prop_name] = rendered

    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(tool["parameters"]["required"]),
        },
    }


_WEEKDAY_ZH = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]


def _weekday_zh_for(ref_date):
    """從 YYYY-MM-DD 算星期（供 weekday=True 模式；與前端 getCurrentDateAnchor 同邏輯）。

    前端用 getDay()（Sunday=0）索引 ['星期日'..]；Python 對應寫法是
    isoweekday() % 7（Monday=1..Saturday=6 → 1..6；Sunday=7 → 0）。
    注意：date.weekday() 是 Monday=0..Sunday=6，直接索引會整體錯位一天，
    不可使用（舊版 bug 曾造成星期日→星期六、星期一→星期日）。
    """
    from datetime import date

    y, m, d = map(int, ref_date.split("-"))
    return _WEEKDAY_ZH[date(y, m, d).isoweekday() % 7]


def build_system_prompt(tool_names, ref_date, enums, weekday=False, compact=False, schema=None):
    """組裝完整 system prompt（位元穩定）。

    Args:
        tool_names: tool 名稱清單（順序 = prompt 中出現順序；單個時為 v1 物件格式）
        ref_date: "YYYY-MM-DD"
        enums: {"category": [...], "account": [...], "contact": [...]}（只含被用到的）
        weekday: True 時前綴加（星期X）
        compact: True 時 JSON 無空格（對齊前端 JSON.stringify；v1 訓練資料為 spaced）
    """
    if schema is None:
        schema = load_schema()

    defs = [render_tool_def(_get_tool(schema, n), enums) for n in tool_names]
    tools_json_obj = defs[0] if len(defs) == 1 else defs
    if compact:
        tools_json = json.dumps(tools_json_obj, ensure_ascii=False, separators=(",", ":"))
    else:
        tools_json = json.dumps(tools_json_obj, ensure_ascii=False)

    weekday_part = f"（{_weekday_zh_for(ref_date)}）" if weekday else ""
    return (
        f"今天是 {ref_date}{weekday_part}。你是一個記帳助理。你被賦予了以下 tools:\n"
        f"{tools_json}"
    )
