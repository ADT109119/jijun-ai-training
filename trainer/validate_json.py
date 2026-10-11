import re
import json
import os

# Protocol v2：schema 路徑（與 gen/build_prompt.py 共用同一真相源）
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools.schema.json")
_schema_cache = None


def load_tools_schema(path=None):
    """載入 tools.schema.json（單一真相源），帶快取。"""
    global _schema_cache
    if path is None and _schema_cache is not None:
        return _schema_cache
    with open(path or SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    if path is None:
        _schema_cache = schema
    return schema


def get_tool_definition(schema, name):
    for tool in schema["tools"]:
        if tool["name"] == name:
            return tool
    raise KeyError(f"未知 tool: {name} (schema 定義: {[t['name'] for t in schema['tools']]})")

def extract_tool_call(response_text):
    """
    對齊 PWA 前端 aiService.js 的正規表示式與容錯解析邏輯，從模型產生的文字中提取 Tool Call。
    支援標準 JSON 格式與特殊標記壓縮格式（雙軌）。
    
    Args:
        response_text (str): 模型輸出的原始文字。
        
    Returns:
        dict: 解析後的 Tool Call 參數字典。
        
    Raises:
        ValueError: 無法提取參數或格式無效。
    """
    if not response_text:
        raise ValueError("模型輸出為空")

    # 0. 剔除思考鏈標籤以相容 Reasoning/Thinking 模型 (如 DeepSeek-R1 / Qwen3 推理系列)
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text)

    # 1. 檢測是否為「特殊標記壓縮格式」 (含有專用特殊 token)
    if "[AMT]" in response_text or "[CAT]" in response_text:
        args = {}
        # 尋找日期
        date_match = re.search(r"\[DATE\]\s*(\d{4}-\d{2}-\d{2})", response_text)
        if date_match:
            args["date"] = date_match.group(1)
        # 尋找金額
        amt_match = re.search(r"\[AMT\]\s*([0-9.]+)", response_text)
        if amt_match:
            try:
                args["amount"] = float(amt_match.group(1))
            except ValueError:
                pass
        
        # 尋找分類
        cat_match = re.search(r"\[CAT\]\s*([^\[<]+)", response_text)
        if cat_match:
            args["category"] = cat_match.group(1).strip()
            
        # 尋找帳戶
        acc_match = re.search(r"\[ACC\]\s*([^\[<]+)", response_text)
        if acc_match:
            args["account"] = acc_match.group(1).strip()
            
        # 尋找備註
        desc_match = re.search(r"\[DESC\]\s*([^\[<]*)", response_text)
        if desc_match:
            args["description"] = desc_match.group(1).strip()
            
        # 尋找收支類型
        type_match = re.search(r"\[TYPE\]\s*([^\[<]+)", response_text)
        if type_match:
            args["type"] = type_match.group(1).strip()
            
        if not args:
            raise ValueError(f"無法從壓縮格式中提取欄位，原始字串為: {response_text}")
        return args

    # 2. 通用 JSON 格式解析
    json_str = ""
    # 優先匹配 <tool_call>...</tool_call> 標籤
    tool_call_match = re.search(r"<tool_call>([\s\S]*?)</tool_call>", response_text)
    if tool_call_match:
        json_str = tool_call_match.group(1).strip()
    else:
        # Fallback: 尋找第一個大括號包裹的區塊
        brace_match = re.search(r"\{[\s\S]*?\}", response_text)
        if brace_match:
            json_str = brace_match.group(0).strip()

    if not json_str:
        raise ValueError("無法從 AI 輸出中提取 Tool Call JSON")

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 失敗: {str(e)}，提取的字串為: {json_str}")

    # 3. 處理包裹在 {"name": "add_record", "args": {...}} 或 {"name": "add_record", "arguments": {...}} 內的情形
    if isinstance(parsed, dict):
        if parsed.get("name") == "add_record":
            if "args" in parsed:
                return parsed["args"]
            if "arguments" in parsed:
                return parsed["arguments"]
        if "args" in parsed:
            return parsed["args"]
        if "arguments" in parsed:
            return parsed["arguments"]
        return parsed
    else:
        raise ValueError("解析後的 JSON 不是物件/字典格式")


# ---------------------------------------------------------------------------
# Protocol v2：多 tool call 解析 + schema 驅動驗證（對齊前端 toolsSchema.js）
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")


def extract_tool_calls(response_text):
    """
    Protocol v2：從模型輸出中提取 1..N 個 tool call。

    對齊前端 aiService.parseToolCalls：
    - 先剔除  思考鏈標籤
    - 壓縮軌（[AMT]/[CAT]）= 單一 add_record（研究用）
    - JSON 軌：全域匹配所有 <tool_call> block；無 tag 時 fallback 第一個 balanced JSON 物件
    - bare-args（無 name envelope）容錯為 add_record

    Returns:
        list[dict]: 每项 {"name": str|None, "args": dict|None,
                         "error": str|None}（error 非 None = 該 block 解析失敗）
    """
    if not response_text:
        raise ValueError("模型輸出為空")
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

    # 1. 壓縮軌（研究用，僅 add_record 單 call）
    if "[AMT]" in response_text or "[CAT]" in response_text:
        try:
            args = extract_tool_call(response_text)
            return [{"name": "add_record", "args": args, "error": None}]
        except ValueError as e:
            return [{"name": "add_record", "args": None, "error": str(e)}]

    calls = []
    # 2. JSON 軌：全域匹配
    for m in _TOOL_CALL_RE.finditer(response_text):
        calls.append(_resolve_call_json(m.group(1).strip()))
    if calls:
        return calls

    # 3. 無 tag fallback：第一個 balanced JSON 物件
    json_str = _extract_balanced_json(response_text)
    if json_str:
        calls.append(_resolve_call_json(json_str))
    if calls:
        return calls

    raise ValueError("無法從 AI 輸出中提取 Tool Call JSON")


def _extract_balanced_json(text):
    """從文字中找出第一座 balanced JSON 物件（含巢狀大括號，字串內大括號不計）。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _resolve_call_json(json_str):
    """把一個 JSON 字串解析成 {"name", "args", "error"}。"""
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"name": None, "args": None, "error": f"解析 JSON 失敗: {e}"}
    if not isinstance(parsed, dict):
        return {"name": None, "args": None, "error": "tool call 不是物件"}
    name = parsed.get("name")
    args = parsed.get("args", parsed.get("arguments"))
    if name is None:
        # bare-args 容錯 → add_record（對齊前端決策）
        if isinstance(args, dict):
            return {"name": "add_record", "args": args, "error": None}
        return {"name": "add_record", "args": parsed, "error": None}
    if not isinstance(args, dict):
        return {"name": name, "args": None, "error": "缺少 args"}
    return {"name": name, "args": args, "error": None}


def validate_tool_call(name, args, pools=None, schema=None):
    """
    Protocol v2：schema 驅動驗證一個 tool call（對齊前端 validateCall）。

    Args:
        name (str): tool 名稱
        args (dict): 參數
        pools (dict, optional): 動態 enum 池 {"category": [...], "account": [...], "contact": [...]}
        schema (dict, optional): tools.schema.json 內容（預設自動載入）

    Returns:
        dict: {"valid": bool, "needs_confirmation": bool, "issues": [str],
               "resolved_from": {field: original_value}}
              valid=False = 靜態檢查失敗（絕不執行）
              needs_confirmation=True = 動態 enum 未命中（UI 讓使用者選）
    """
    if schema is None:
        schema = load_tools_schema()
    try:
        tool = get_tool_definition(schema, name)
    except KeyError as e:
        return {"valid": False, "needs_confirmation": False, "issues": [str(e)], "resolved_from": {}}
    if not isinstance(args, dict):
        return {"valid": False, "needs_confirmation": False, "issues": ["args 不是物件"], "resolved_from": {}}

    properties = tool["parameters"]["properties"]
    required = tool["parameters"]["required"]
    validation = tool.get("validation", {})
    runtime_optional = set(validation.get("runtime_optional", []))
    pools = pools or {}
    issues = []
    resolved_from = {}

    # 1. 必填（訓練資料必填，但 runtime_optional 欄位運行時允許缺省）
    for key in required:
        if key in runtime_optional:
            continue
        if key not in args or args[key] in (None, ""):
            issues.append(f"缺少必要欄位: {key}")

    # 2. 未預期欄位
    extras = [k for k in args if k not in properties]
    if extras:
        issues.append(f"未預期欄位: {', '.join(extras)}")

    # 3. 逐欄位型別 / enum / 範圍 / 日期
    for key, prop in properties.items():
        value = args.get(key)
        if value in (None, ""):
            continue
        ptype = prop.get("type")
        if ptype in ("number", "integer"):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                issues.append(f"欄位 {key} 必須是數值")
                continue
            if ptype == "integer" and not float(value).is_integer():
                issues.append(f"欄位 {key} 必須是整數")
                continue
            if prop.get("minimum") is not None and value < prop["minimum"]:
                issues.append(f"欄位 {key} 必須 ≥ {prop['minimum']}")
            if prop.get("maximum") is not None and value > prop["maximum"]:
                issues.append(f"欄位 {key} 必須 ≤ {prop['maximum']}")
            continue
        if ptype == "string":
            if not isinstance(value, str):
                issues.append(f"欄位 {key} 必須是字串")
                continue
            if "enum_source" in prop:
                pool = pools.get(prop["enum_source"], [])
                if pool and value not in pool:
                    matched = _fuzzy_match(value, pool)
                    if matched is not None:
                        args[key] = matched
                        resolved_from[key] = value
                    else:
                        issues.append(f"needs_confirmation:{key}={value}")
            elif "enum" in prop and value not in prop["enum"]:
                issues.append(f"欄位 {key} 不在合法清單: {value}")
            if key == "date" and not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                issues.append(f"日期格式無效: {value} (必須為 YYYY-MM-DD)")

    # 4. validation 區塊
    if validation.get("amount") == "gt0" and isinstance(args.get("amount"), (int, float)) and not isinstance(args.get("amount"), bool):
        if not args["amount"] > 0:
            issues.append("金額必須大於 0")
    for rule in validation.get("custom", []):
        if rule == "from_account != to_account" and args.get("from_account") is not None \
                and args.get("from_account") == args.get("to_account"):
            issues.append("轉帳來源與目標帳戶相同")
    for rule in validation.get("conditional", []):
        # 支援結構化規則 {"when": {...}, "required": [...], "message": str}
        if isinstance(rule, dict):
            when = rule.get("when", {})
            if all(args.get(k) == v for k, v in when.items()):
                for req in rule.get("required", []):
                    if args.get(req) in (None, ""):
                        issues.append(rule.get("message", f"缺少欄位: {req}"))

    needs_confirmation = any(i.startswith("needs_confirmation:") for i in issues)
    valid = not any(not i.startswith("needs_confirmation:") for i in issues)
    return {
        "valid": valid,
        "needs_confirmation": needs_confirmation,
        "issues": [i.replace("needs_confirmation:", "") for i in issues],
        "resolved_from": resolved_from,
    }


def _normalize_enum(s):
    return re.sub(r"\s+", "", str(s).strip().lower())


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i]
        for j in range(1, len(b) + 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)))
        prev = curr
    return prev[len(b)]


def _fuzzy_match(value, pool):
    """模糊匹配動態 enum（normalization → 子字串 → 編輯距離 ≤ 1），對齊前端 fuzzyMatch。"""
    nv = _normalize_enum(value)
    if not nv:
        return None
    for p in pool:
        if _normalize_enum(p) == nv:
            return p
    for p in pool:
        np_ = _normalize_enum(p)
        if len(np_) >= 2 and (np_ in nv or nv in np_):
            return p
    best, best_dist = None, 2
    for p in pool:
        d = _levenshtein(nv, _normalize_enum(p))
        if d < best_dist:
            best, best_dist = p, d
    return best if best_dist <= 1 else None


def validate_record(args, categories=None, accounts=None):
    """
    驗證提取的參數是否符合記帳規則與 schema。
    
    Args:
        args (dict): 提取出的參數字典。
        categories (list, optional): 合法的分類列表 (用於 dynamic enum 檢查)。
        accounts (list, optional): 合法的帳戶列表 (用於 dynamic enum 檢查)。
        
    Returns:
        tuple: (bool, str) - 是否通過驗證，以及錯誤說明。
    """
    # 1. 檢查必要欄位
    required_fields = ["amount", "category", "account", "type"]
    for field in required_fields:
        if field not in args:
            return False, f"缺少必要欄位: {field}"

    # 2. 驗證金額
    try:
        amount = float(args["amount"])
        if amount <= 0:
            return False, "金額必須大於 0"
    except (ValueError, TypeError):
        return False, "金額非有效數值"

    # 3. 驗證記帳類型
    if args["type"] not in ["expense", "income"]:
        return False, f"無效的記帳類型: {args['type']} (必須為 expense 或 income)"

    # 4. 驗證分類是否在動態清單中 (選填)
    if categories is not None and args["category"] not in categories:
        return False, f"分類 '{args['category']}' 不在合法分類清單中"

    # 5. 驗證帳戶是否在動態清單中 (選填)
    if accounts is not None and args["account"] not in accounts:
        return False, f"帳戶 '{args['account']}' 不在合法帳戶清單中"

    # 6. 驗證日期格式 (選填)
    if "date" in args and args["date"]:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(args["date"])):
            return False, f"日期格式無效: {args['date']} (必須為 YYYY-MM-DD)"

    return True, ""

def calculate_reward(response_text, label_dict, categories=None, accounts=None):
    """
    計算用於強化學習（例如 GRPO / PPO）的 Reward 值。
    依據格式相容度、Schema 合規性與數值精準度給分。
    
    Args:
        response_text (str): 模型產生的字串。
        label_dict (dict): 預期正確的參數字典 (Label)。
        categories (list, optional): 動態分類清單。
        accounts (list, optional): 動態帳戶清單。
        
    Returns:
        float: Reward 分數。最低 -1.0，最高 +4.0。
    """
    reward = 0.0

    # 1. 格式與 JSON 提取驗證
    try:
        args = extract_tool_call(response_text)
        reward += 1.0  # JSON 提取成功，給予基礎分
    except ValueError:
        return -1.0    # 格式毀損，直接給予嚴重懲罰分

    # 2. Schema 欄位與型態校驗
    is_valid, _ = validate_record(args, categories, accounts)
    if is_valid:
        reward += 1.0  # 欄位齊全且類型正確，再加 1.0 分
    else:
        return 0.0     # 雖然是 JSON，但內容不合規，止步於 1.0 分（總分 0+0）

    # 3. 欄位數值完全一致性比對 (與 Label 比對)
    exact_match_fields = ["amount", "category", "account", "type"]
    match_count = 0
    for field in exact_match_fields:
        if field == "amount":
            # 浮點數容錯比對
            try:
                if abs(float(args.get("amount", 0)) - float(label_dict.get("amount", 0))) < 1e-4:
                    match_count += 1
            except (ValueError, TypeError):
                pass
        else:
            if args.get(field) == label_dict.get(field):
                match_count += 1

    # 若 4 個關鍵欄位完全符合 label，加滿 2.0 分；否則依比例給分
    reward += (match_count / 4.0) * 2.0

    return reward

if __name__ == "__main__":
    # 測試程式碼
    mock_model_output = '<tool_call>{"name": "add_record", "args": {"amount": 150, "category": "餐飲", "account": "信用卡", "description": "午餐", "type": "expense"}}</tool_call>'
    mock_label = {"amount": 150, "category": "餐飲", "account": "信用卡", "type": "expense"}
    
    print("提取結果:", extract_tool_call(mock_model_output))
    print("校驗結果:", validate_record(extract_tool_call(mock_model_output)))
    print("RL Reward 分數 (預期 4.0):", calculate_reward(mock_model_output, mock_label))
