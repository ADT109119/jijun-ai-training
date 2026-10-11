"""
gen/gen_js_tools.py — 把 tools.schema.json 同步到前端 src/js/toolsSchema.js 的標記區段

用法：
    python3 gen/gen_js_tools.py            # 同步（改寫標記區段）
    python3 gen/gen_js_tools.py --check    # 只檢查是否漂移（CI 用，漂移時 exit 1）

標記區段：
    // BEGIN GENERATED FROM tools.schema.json (gen/gen_js_tools.py) — DO NOT EDIT MANUALLY
    ...
    // END GENERATED FROM tools.schema.json

注意：add_record.date 的 description 前端用「ISO 8601 格式日期，例如 YYYY-MM-DD」
（v1 凍結位元，無「僅日期精度」四字）；JS 覆寫表 FRONTEND_OVERRIDES 處理此差異，
位元差異記錄於 docs/p0-protocol-v2-spec.md §1。
"""

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
SCHEMA_PATH = os.path.join(ROOT, "tools.schema.json")
JS_PATH = os.path.join(ROOT, "..", "..", "src", "js", "toolsSchema.js")

BEGIN = "// BEGIN GENERATED FROM tools.schema.json (gen/gen_js_tools.py) — DO NOT EDIT MANUALLY"
END = "// END GENERATED FROM tools.schema.json"

# 前端覆寫：v1 凍結位元（add_record.date description 無「僅日期精度」）
FRONTEND_OVERRIDES = {
    "add_record": {
        "date": {
            "description": "ISO 8601 格式日期，例如 YYYY-MM-DD"
        }
    }
}


def _to_js(value, indent=0):
    """把 Python 結構轉成 JS 物件字面量（2 空格縮排，保持欄位順序）。"""
    pad = "    " * indent
    pad2 = "    " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = [f"{pad}{{"]
        for k, v in value.items():
            key = k if re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", k) else f'"{k}"'
            lines.append(f"{pad2}{key}: {_to_js(v, indent + 1)},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ", ".join(_to_js(v, indent) for v in value)
        return f"[{inner}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False)


def build_js_block(schema):
    """產出標記區段內部的 JS 內容（不含 BEGIN/END 行）。"""
    tools_js = []
    for tool in schema["tools"]:
        t = {
            "name": tool["name"],
            "description": tool["description"],
            "phase": tool["phase"],
            "kind": tool["kind"],
        }
        params = dict(tool["parameters"])
        props = {}
        for k, v in tool["parameters"]["properties"].items():
            props[k] = dict(v)
            override = FRONTEND_OVERRIDES.get(tool["name"], {}).get(k)
            if override:
                props[k] = {**props[k], **override}
        params["properties"] = props
        t["parameters"] = params
        if "validation" in tool:
            t["validation"] = tool["validation"]
        tools_js.append(t)

    lines = ["    tools: ["]
    for i, t in enumerate(tools_js):
        body = _to_js(t, indent=1)
        if i < len(tools_js) - 1:
            lines.append(body.rstrip() + ",")
        else:
            lines.append(body.rstrip())
    lines.append("    ],")
    # contexts：只保留「情境 → tools 清單」映射（剔除 note 等文件欄位）
    contexts = {k: v for k, v in schema["contexts"].items() if isinstance(v, list)}
    lines.append(f"    contexts: {_to_js(contexts)},")
    # dynamicEnumSources：JS 端只需 maxItems（snake_case → camelCase，剔除訓練端欄位）
    des = {}
    for k, v in schema["dynamic_enum_sources"].items():
        entry = {}
        if "max_items" in v:
            entry["maxItems"] = v["max_items"]
        des[k] = entry
    lines.append(f"    dynamicEnumSources: {_to_js(des)}")
    return "\n".join(lines)


def main():
    check_only = "--check" in sys.argv
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    js_path = os.path.normpath(JS_PATH)
    with open(js_path, "r", encoding="utf-8") as f:
        js_src = f.read()

    m_begin = re.search(re.escape(BEGIN) + r"\n", js_src)
    m_end = re.search(r"\n([ \t]*)" + re.escape(END), js_src)
    if not m_begin or not m_end:
        print("[FAIL] toolsSchema.js 找不到標記區段")
        sys.exit(1)

    new_block = build_js_block(schema)
    old_block = js_src[m_begin.end():m_end.start()]

    if old_block == new_block:
        print("[OK] toolsSchema.js 與 tools.schema.json 同步，無漂移")
        return

    if check_only:
        print("[DRIFT] toolsSchema.js 標記區段與 tools.schema.json 不同步！")
        print("       執行: python3 gen/gen_js_tools.py 同步")
        sys.exit(1)

    js_src = js_src[:m_begin.end()] + new_block + js_src[m_end.start():]
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_src)
    print(f"[OK] 已同步 tools.schema.json → {os.path.relpath(js_path)}")


if __name__ == "__main__":
    main()
