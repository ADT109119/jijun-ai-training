import re
import json

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

    # 3. 處理包裹在 {"name": "add_record", "args": {...}} 內的情形
    if isinstance(parsed, dict):
        if parsed.get("name") == "add_record" and "args" in parsed:
            return parsed["args"]
        elif "args" in parsed:
            return parsed["args"]
        return parsed
    else:
        raise ValueError("解析後的 JSON 不是物件/字典格式")

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
