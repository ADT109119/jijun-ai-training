import os
import json
import random
import asyncio
import argparse
from datetime import date, timedelta
from openai import AsyncOpenAI, APIError, RateLimitError
from trainer.validate_json import extract_tool_call, validate_record

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import re

def _normalize_sample(data):
    """將各種 JSON 格式統一為 {'difficulty_level':..., 'messages':[...]} 格式"""
    if isinstance(data, dict):
        if "messages" in data:
            # 進一步檢查 assistant content 是否缺 tool_call，若有 name+args 則自動組裝
            for i, msg in enumerate(data["messages"]):
                if msg.get("role") == "assistant":
                    c = msg.get("content", "")
                    if (not c or "<tool_call>" not in c) and "name" in msg and "args" in msg:
                        tc = json.dumps({"name": msg["name"], "args": msg["args"]}, ensure_ascii=False)
                        msg["content"] = f"<tool_call>{tc}</tool_call>"
            return data
        # flat format: {"system":..., "user":..., "assistant":...}
        if "system" in data and "user" in data and "assistant" in data:
            return {
                "difficulty_level": data.get("difficulty_level", "Level-1 (Simple)"),
                "messages": [
                    {"role": "system", "content": data["system"]},
                    {"role": "user", "content": data["user"]},
                    {"role": "assistant", "content": data["assistant"]}
                ]
            }
        if "difficulty_level" in data:
            sys_ = data.get("system", "")
            usr_ = data.get("user", "")
            asst_ = data.get("assistant", "")
            msgs_ = []
            if sys_:
                msgs_.append({"role": "system", "content": sys_})
            if usr_:
                msgs_.append({"role": "user", "content": usr_})
            if asst_:
                msgs_.append({"role": "assistant", "content": asst_})
            if msgs_:
                data["messages"] = msgs_
                return data
    if isinstance(data, list):
        # array of message objects
        messages = []
        difficulty = "Level-1 (Simple)"
        for item in data:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role in ("system", "user", "assistant"):
                    if not content and "name" in item and "args" in item:
                        tc = json.dumps({"name": item["name"], "args": item["args"]}, ensure_ascii=False)
                        content = f"<tool_call>{tc}</tool_call>"
                    messages.append({"role": role, "content": content})
                if "difficulty_level" in item:
                    difficulty = item["difficulty_level"]
        if len(messages) >= 2:
            return {"difficulty_level": difficulty, "messages": messages}
    return None

def _fix_unescaped_quotes_in_content(text: str) -> str:
    """嘗試修復 assistant content 中 JSON-in-JSON 的未跳脫引號問題。"""
    import re
    # 搜尋 pattern: "content": "<tool_call>{...內容可能包含未跳脫的引號...}</tool_call>"
    # 策略：找到 content 欄位後，將其內容中的所有未跳脫 " 取代為 \"
    in_content = False
    content_start = -1
    result = list(text)
    i = 0
    while i < len(text):
        if not in_content:
            # 尋找 "content": 後接 " 的位置
            m = re.search(r'"(?:content|assistant)"\s*:\s*"', text[i:])
            if m:
                # 這個 " 開始了 content 的內容
                content_start = i + m.end() - 1  # 指向開啟的 "
                in_content = True
                i = content_start + 1
                continue
            break
        else:
            # 在 content 內部，尋找未跳脫的結束 "
            ch = text[i]
            if ch == '\\':
                i += 2  # 跳過跳脫序列
                continue
            if ch == '"':
                # 檢查這是否真的是 content 的結束（後面是 </tool_call>"）
                rest = text[i:]
                if rest.startswith('</tool_call>'):
                    # 找到結束，跳出
                    in_content = False
                    i += len('</tool_call>')
                    continue
                # 檢查是否前面有 <tool_call>（即這是 content 內的第一個 "）
                before = text[content_start:i]
                if '<tool_call>{' in before or '<tool_call>' in before:
                    # 這是內層 JSON 的引號，需要跳脫
                    result[i] = '\\"'
                    i += 1
                    continue
            i += 1
    return ''.join(result)

def clean_and_extract_json(raw_text: str):
    if not raw_text or not raw_text.strip():
        raise ValueError("模型返回內容為空 (raw_content is empty)")

    text = raw_text.strip()

    # 1. 剔除思考鏈標籤
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # 2. 匹配 ```json ... ``` 包裹的內容
    for matches in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        block = matches.group(1).strip()
        try:
            data = json.loads(block, strict=False)
            normalized = _normalize_sample(data)
            if normalized:
                return normalized
        except Exception:
            pass

    # 3. 字串感知的平衡大括號 {...} 區塊提取
    brace_candidates = []
    in_string = False
    escape = False
    stack = []
    start_idx = -1

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\':
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        brace_candidates.append(text[start_idx:i+1])

    for candidate in brace_candidates:
        try:
            data = json.loads(candidate, strict=False)
            normalized = _normalize_sample(data)
            if normalized:
                return normalized
        except Exception:
            pass

    # 4. 嘗試提取 JSON 陣列 (以 [ 開頭)
    if text.strip().startswith("["):
        for matches in re.finditer(r"(\[[\s\S]*?\])", text):
            block = matches.group(1).strip()
            try:
                data = json.loads(block, strict=False)
                normalized = _normalize_sample(data)
                if normalized:
                    return normalized
            except Exception:
                pass

    # 5. Fallback: 嘗試直接對全文做 json.loads
    try:
        data = json.loads(text, strict=False)
        normalized = _normalize_sample(data)
        if normalized:
            return normalized
    except Exception:
        pass

    # 6. 嘗試使用修復函數處理常見的 JSON-in-JSON 引號問題後再解析
    try:
        fixed = _fix_unescaped_quotes_in_content(text)
        if fixed != text:
            for matches in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", fixed, re.IGNORECASE):
                block = matches.group(1).strip()
                try:
                    data = json.loads(block, strict=False)
                    normalized = _normalize_sample(data)
                    if normalized:
                        return normalized
                except Exception:
                    pass
            data = json.loads(fixed, strict=False)
            normalized = _normalize_sample(data)
            if normalized:
                return normalized
    except Exception:
        pass

    preview = text[:200] + "..." if len(text) > 200 else text
    raise ValueError(f"無法在模型回應中提取出合法的 JSON 物件。模型原始輸出片段為:\n{preview}")


# 擴充更豐富的預設分類與帳戶，增加多樣性
CATEGORIES = [
    "飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款",
    "薪水", "獎金", "零用錢", "兼職", "投資", "利息", "欠款回收",
    "其他"
]

EXPENSE_CATS = {"飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款", "其他"}
INCOME_CATS = {"薪水", "獎金", "零用錢", "兼職", "投資", "利息", "欠款回收", "其他"}

ACCOUNTS = [
    # 現金
    "現金",
    # 電子票證
    "悠遊卡", "一卡通", "icash", "有錢卡",
    # 銀行數位帳戶
    "Richart", "LINE Bank", "將來銀行", "樂天銀行", "O-Bank",
    # 行動支付
    "街口支付", "LINE Pay", "Apple Pay", "Google Pay", "Samsung Pay",
    "台灣Pay", "全盈支付", "全支付", "OPEN錢包", "Pi錢包",
    "橘子支付", "歐付寶", "悠遊付",
    # 信用卡 (銀行)
    "台新信用卡", "國泰信用卡", "富邦信用卡", "中信信用卡",
    "玉山信用卡", "永豐信用卡", "聯邦信用卡", "元大信用卡",
    "星展信用卡", "滙豐信用卡", "兆豐信用卡", "樂天信用卡",
    # 銀行存款 / 帳戶
    "郵局帳戶", "台新帳戶", "國泰帳戶", "富邦帳戶",
    "中信帳戶", "玉山帳戶", "永豐帳戶", "兆豐帳戶",
    "第一帳戶", "華南帳戶", "彰化帳戶", "合庫帳戶",
    "土地銀行帳戶", "台灣銀行帳戶", "元大帳戶", "渣打帳戶",
    # 子帳戶 / 數位帳戶
    "國金子帳戶", "Richart子帳戶", "大戶DAWHO", "iLeo帳戶",
    "KOKO帳戶", "NBA帳戶", "SnY帳戶",
    # 外幣 / 投資
    "外幣帳戶", "證券交割戶", "複委託帳戶", "數位外幣帳戶",
    # 其他
    "銀行存款", "加密貨幣", "美元帳戶", "日圓帳戶",
]

EXTRA_CATEGORIES_POOL = [
    # === 支出 (expense) ===
    # 飲食分支
    ("手搖飲", "expense"), ("咖啡", "expense"), ("麵包甜點", "expense"),
    ("小吃", "expense"), ("早午餐", "expense"), ("宵夜", "expense"),
    ("外送", "expense"), ("聚餐", "expense"), ("飲料", "expense"),
    ("便當", "expense"),
    # 日常 / 居家
    ("水費", "expense"), ("電費", "expense"), ("瓦斯費", "expense"),
    ("管理費", "expense"), ("房租", "expense"), ("房貸", "expense"),
    ("清潔用品", "expense"), ("家用", "expense"), ("裝潢", "expense"),
    ("園藝", "expense"), ("廚具", "expense"), ("寢具", "expense"),
    ("家居", "expense"), ("修繕", "expense"),
    # 交通分支
    ("停車費", "expense"), ("油費", "expense"), ("罰單", "expense"),
    ("過路費", "expense"), ("保養", "expense"), ("輪胎", "expense"),
    ("機車", "expense"), ("汽車", "expense"), ("自行車", "expense"),
    ("維修", "expense"),
    # 娛樂 / 休閒
    ("電影", "expense"), ("音樂", "expense"), ("演唱會", "expense"),
    ("展覽", "expense"), ("舞台劇", "expense"), ("KTV", "expense"),
    ("酒吧", "expense"), ("夜店", "expense"), ("酒店", "expense"),
    ("旅行", "expense"), ("飯店", "expense"), ("機票", "expense"),
    ("住宿", "expense"), ("露營", "expense"), ("郵輪", "expense"),
    ("按摩", "expense"), ("SPA", "expense"),
    # 購物
    ("網購", "expense"), ("百貨", "expense"), ("超市", "expense"),
    ("超商", "expense"), ("藥妝", "expense"), ("菜市場", "expense"),
    ("夜市", "expense"), ("Costco", "expense"), ("全聯", "expense"),
    # 3C / 通訊
    ("手機", "expense"), ("電腦", "expense"), ("軟體", "expense"),
    ("雲端服務", "expense"), ("網路費", "expense"), ("手機費", "expense"),
    ("配件", "expense"), ("耳機", "expense"), ("智慧手錶", "expense"),
    ("相機", "expense"), ("遊戲主機", "expense"),
    # 服飾美容
    ("衣服", "expense"), ("鞋子", "expense"), ("包包", "expense"),
    ("飾品", "expense"), ("手錶", "expense"), ("內衣", "expense"),
    ("泳衣", "expense"), ("帽子", "expense"), ("西裝", "expense"),
    ("保養品", "expense"), ("化妝品", "expense"), ("香水", "expense"),
    ("理髮", "expense"), ("美髮", "expense"), ("美容", "expense"),
    ("美甲", "expense"), ("醫美", "expense"),
    # 醫療保健
    ("牙醫", "expense"), ("中醫", "expense"), ("眼科", "expense"),
    ("皮膚科", "expense"), ("藥品", "expense"), ("健檢", "expense"),
    ("手術", "expense"), ("住院", "expense"), ("復健", "expense"),
    ("眼鏡", "expense"), ("隱眼", "expense"),
    ("保健食品", "expense"), ("口罩", "expense"),
    # 教育 / 學習
    ("補習班", "expense"), ("家教", "expense"), ("線上課程", "expense"),
    ("書籍", "expense"), ("雜誌", "expense"), ("教材", "expense"),
    ("文具用品", "expense"), ("證照考試", "expense"), ("研討會", "expense"),
    # 寵物
    ("寵物美容", "expense"), ("寵物醫療", "expense"), ("寵物食品", "expense"),
    ("寵物用品", "expense"), ("寵物寄宿", "expense"),
    # 其他支出
    ("禮物", "expense"), ("交際", "expense"), ("應酬", "expense"),
    ("喜宴", "expense"), ("白包", "expense"), ("稅金", "expense"),
    ("捐款", "expense"), ("保險", "expense"), ("健身", "expense"),
    ("運動用品", "expense"), ("運動器材", "expense"), ("球鞋", "expense"),
    ("高爾夫", "expense"), ("游泳", "expense"), ("羽毛球", "expense"),
    ("菸酒", "expense"), ("紅包", "expense"), ("罰款", "expense"),
    ("運費", "expense"), ("手續費", "expense"), ("服務費", "expense"),
    # === 收入 (income) ===
    ("年終", "income"), ("三節獎金", "income"), ("績效獎金", "income"),
    ("業績獎金", "income"), ("尾牙", "income"), ("抽獎", "income"),
    ("彩券", "income"), ("發票中獎", "income"),
    ("退稅", "income"), ("保險理賠", "income"), ("理賠金", "income"),
    ("遺產", "income"), ("贈與", "income"),
    ("獎學金", "income"), ("助學金", "income"), ("補助金", "income"),
    ("育兒津貼", "income"), ("生育補助", "income"),
    ("退休金", "income"), ("勞退", "income"), ("國民年金", "income"),
    ("股息", "income"), ("配息", "income"), ("基金收益", "income"),
    ("債券利息", "income"),
    ("稿費", "income"), ("演講費", "income"), ("顧問費", "income"),
    ("仲介費", "income"), ("推薦獎金", "income"),
    ("回饋金", "income"), ("點數回饋", "income"), ("刷卡回饋", "income"),
    ("二手販賣", "income"), ("網拍", "income"), ("代購", "income"),
    ("小費", "income"), ("跑腿費", "income"),
    ("廣告收入", "income"), ("聯盟行銷", "income"),
    ("YT收益", "income"), ("直播收益", "income"), ("創作者基金", "income"),
    ("贊助", "income"), ("業配", "income"),
    ("版稅", "income"), ("專利收入", "income"), ("授權費", "income"),
    ("教學收入", "income"), ("教練費", "income"), ("諮詢費", "income"),
    ("翻譯費", "income"), ("設計費", "income"), ("攝影費", "income"),
    ("接案", "income"), ("外包", "income"), ("代辦費", "income"),
    # （重複已存在的在生成時不會有影響，因為我們用 random.sample 從 pool 中取）
]

# 新增隨機情境池，打破 LLM 單一邏輯複製，增強數據分佈多樣性
SCENARIOS = [
    "上班族午餐吃便當或商業午餐，或買咖啡飲料",
    "晚餐去餐廳吃飯或叫外送",
    "早餐在早餐店買三明治飲料",
    "去超市或便利商店買日常用品、零食飲料",
    "繳水電費、瓦斯費、管理費或停車費",
    "搭捷運、公車或計程車通勤",
    "騎機車加油或搭高鐵出差",
    "看電影、去遊樂園或買遊戲",
    "去診所看醫生或去藥局買藥",
    "掛號費、健檢費用或買保健食品",
    "買書、報名課程或繳學費",
    "參加研討會或買線上課程",
    "幫朋友代墊費用後對方還錢",
    "收到朋友還款或退款",
    "收到兼職收入或打工薪水",
    "買衣服、鞋子或網購日常用品",
    "買寵物用品或幫寵物看醫生",
    "繳保險費或繳房租房貸",
    "買家電、家具或修繕房屋",
    "收到股利、利息或賣二手物品收入",
    "領到這個月的薪水或年終獎金",
    "爸媽給的零用錢或生活費",
    "兼差接案或打工賺的外快",
    "股票配息或銀行利息收入",
    "朋友還之前借的錢或欠款回收"
]

# 新增口語化風格池，確保對話句式與長度分佈豐富
LANG_STYLES = [
    "極簡短句，甚至省略主詞或動詞（例：『午餐吃麵 120 現金』）",
    "囉唆、帶有許多心情故事與情境碎碎念的長句（例：『今天下大雨真煩，下班忍不住去全聯大買特買零食，不知不覺花了我五百多塊，刷了信用卡，心在痛』）",
    "倒裝句或順序混亂的表達（例：『刷了 LINE Pay 買星巴克，花了 160 塊今天早上』）",
    "包含數字或貨幣口語的寫法（例：『去屈臣氏買個洗面乳，花了一張藍色小朋友，找回的零錢用悠遊卡嗶了』）",
    "台灣在地日常用語與語助詞（例：『哇賽，剛剛去美聯社買牛奶，用悠遊卡扣了 90 塊耶』）",
    "帶有明確日期指代（例：『上禮拜三去健身房扣款 1200，刷了台新卡』或『昨天領了外包薪水 3 萬存入銀行』）",
    "包含部分英文術語或簡寫的夾雜口語（例：『買了 Udemy 課程花了 50 USD，刷 Visa 卡』）",
    "帶有大量重複贅詞的口語（例：『阿那個就是…就是…那個午餐的錢啦，好像大概150元吧』）",
    "中英夾雜的科技業用語（例：『今天下午team building去吃了個buffet，我直接用Apple Pay刷了大概890』）",
    "用社群語氣與網路用語（例：『今天掛號費650尻了一筆醫藥費，現金結帳』）",
    "帶有長輩口吻的老派說法（例：『今兒個去市場買了點兒菜，花了二百五，給的是現金』）",
    "用疑問或反問形式記帳（例：『我昨天是不是刷了信用卡買書？好像是400多塊？』）"
]

# 系統 Prompt 與 Tools 定義，用於教導大模型如何生成樣本
# 我們特別在結尾強調了 "JSON" 這個字，以相容各類大模型強制 JSON 輸出的約束
GENERATOR_SYSTEM_PROMPT = """
你是一個專門生成機器學習訓練數據集的 AI 助手。你的任務是批量產生「中文口語記帳與 Tool Call 對話軌跡」的訓練資料。

我們有一個名為 add_record 的記帳工具 (tool)，其參數結構如下:
- amount: 數值，交易金額 (必須 > 0，請用 realistic 的金額：支出 50~5000 元，收入 1000~200000 元)
- category: 字串，交易分類 (必須從給定的分類清單中選擇)
- account: 字串，支付媒介 (必須從給定的帳戶清單中選擇)
- description: 字串，消費的具體說明備註
- type: 字串，必須為 "expense" (支出) 或 "income" (收入)
- date: 字串，交易日期 (必須符合 YYYY-MM-DD 格式，以 {ref_date} 為今日基準計算)

你每次需要根據我指定的「難度等級」和給定的「分類清單、帳戶清單」，產生 1 筆對話資料。
輸出格式必須嚴格為以下 JSON 結構（不要包裝在陣列中，不要加 Markdown 圍欄）：
{
  "difficulty_level": "Level-1 (Simple)",
  "messages": [
    {
      "role": "system",
      "content": "今天是 {ref_date}。你是一個記帳助理。"
    },
    {
      "role": "user",
      "content": "剛剛吃午餐花了 150 元付現"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\\"name\\": \\"add_record\\", \\"args\\": {\\"amount\\": 150, \\"category\\": \\"飲食\\", \\"account\\": \\"現金\\", \\"description\\": \\"午餐\\", \\"type\\": \\"expense\\", \\"date\\": \\"{ref_date}\\"}}</tool_call>"
    }
  ]
}

請確保：
1. user 的內容必須極為日常、口語化，符合台灣繁體中文的用語習慣。
2. 根據不同的難度等級設計對話：
   - Level-1 (Simple)：直白簡單，無雜訊（例：「剛剛吃午餐花了 150 元付現」）。
   - Level-2 (Noise)：加入日常閒聊與修飾字詞，但金額與分類仍明確。
   - Level-3 (Reasoning)：需要簡單推理或時間計算（例：「昨天買的三本小說今天送到了」，模型需推算日期為昨天，且將小說歸類為教育或娛樂）。
3. 使用者的日常口語記帳輸入如果有提及日期、相對時間（例如：今天、昨天、前天、上週三、上個月等），你必須以系統設定的參考日期 {ref_date} 為基準點計算出正確的 date 欄位值（格式為 YYYY-MM-DD）。如果使用者沒有提及任何時間，則將 date 設為 '{ref_date}'（今天）。
4. 輸出的 assistant 欄位中，JSON 的 key 與 value 必須完全合規。其中 args 必須包含 date 欄位。
5. 提供的分類清單與帳戶清單中可能有部分分類在語意上與實際消費場景不符，請你按照分類名稱的字面語意，給出最合理、最貼近該實際消費的分類標籤。
6. 請使用 "args" 作為 key，不要使用 "arguments"。
7. 禁止使用任何 emoji 或顏文字（如 ☁️、😊、QQ 等），僅使用純文字。
"""

async def generate_single_sample(client, model_name, difficulty, categories, accounts, ref_date, recent_history=None, thinking_budget=0, bias_categories=None):
    """
    發送非同步請求至 OpenAI/LLM API 生成單筆樣本，包含自動自我修正 (Self-Correction) 迴圈與後置代碼排重
    """
    sub_categories = list(categories)

    extra_cats = random.sample(EXTRA_CATEGORIES_POOL, k=random.randint(1, 2))
    local_expense_cats = set(EXPENSE_CATS)
    local_income_cats = set(INCOME_CATS)
    for ec_name, ec_type in extra_cats:
        sub_categories.append(ec_name)
        if ec_type == "expense":
            local_expense_cats.add(ec_name)
        else:
            local_income_cats.add(ec_name)

    sub_accounts = ["現金"] + random.sample([a for a in accounts if a != "現金"], k=random.randint(3, 5))

    if bias_categories:
        avail = [c for c in bias_categories if c in sub_categories]
        if avail and random.random() < 0.7:
            target_category = random.choice(avail)
        else:
            target_category = random.choice(sub_categories)
    else:
        target_category = random.choice(sub_categories)
    target_account = random.choice(sub_accounts)

    tool_definition = {
        "name": "add_record",
        "description": "新增一筆記帳記錄",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": list(sub_categories)},
                "account": {"type": "string", "enum": list(sub_accounts)},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": ["expense", "income"]},
                "date": {"type": "string", "description": "ISO 8601 格式日期，僅日期精度，例如 YYYY-MM-DD"}
            },
            "required": ["amount", "category", "account", "type", "date"]
        }
    }

    scenario = random.choice(SCENARIOS)
    lang_style = random.choice(LANG_STYLES)

    ref_date_str = ref_date.isoformat()
    system_prompt = GENERATOR_SYSTEM_PROMPT.replace("{ref_date}", ref_date_str)

    is_income = target_category in local_income_cats
    expected_type = "income" if is_income else "expense"

    newline = chr(10)
    prompt = (
        f"請生成 1 筆包含 <tool_call> 的對話樣本。{newline}"
        f"難度等級: {difficulty}{newline}"
        f"可用的分類清單: {sub_categories}{newline}"
        f"可用的帳戶清單: {sub_accounts}{newline}"
        f"參考日期 (今天): {ref_date_str}{newline}"
        f"請確保對話涉及的分類為 '{target_category}'，帳戶為 '{target_account}'。{newline}"
        f"請注意，分類 '{target_category}' 應使用 type='{expected_type}'，切勿混用。{newline}"
        f"【多樣性要求】{newline}"
        f"1. 必須圍繞以下情境展開故事：{scenario}{newline}"
        f"2. 口語風格限制：{lang_style}{newline}{newline}"
        f"【輸出格式要求】{newline}"
        f"- 請輸出一個 JSON 物件，包含 difficulty_level 和 messages 兩個欄位。{newline}"
        f"- 請勿將 JSON 包裝在陣列中。{newline}"
        f"- messages 是一個陣列，包含 system、user、assistant 三筆物件。{newline}"
        f"- system: '今天是 {ref_date_str}。你是一個記帳助理。'{newline}"
        f"- assistant 的 content 欄位請輸出「純 JSON 字串」，不要包含 <tool_call>、</tool_call> 或任何 XML 標籤。{newline}"
        f"- content 欄位範例（直接輸出此 JSON 字串，不需外框引號）：{newline}"
        f"- {{\"name\": \"add_record\", \"args\": {{\"amount\": 150, \"category\": \"{target_category}\", \"account\": \"{target_account}\", \"description\": \"消費說明\", \"type\": \"{expected_type}\", \"date\": \"{ref_date_str}\"}}}}{newline}"
        f"- 使用 'args' 而非 'arguments'。{newline}"
        f"- 禁止使用 emoji 或顏文字，僅限純文字。{newline}"
        f"- 只能使用分類清單與帳戶清單中的值。{newline}"
    )

    max_retries = 3
    history_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        raw_content = ""
        try:
            extra_body = None
            if "qwen" in model_name.lower():
                if thinking_budget and thinking_budget > 0:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": thinking_budget}}
                else:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            response = await client.chat.completions.create(
                model=model_name,
                messages=history_messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
                extra_body=extra_body
            )
            
            raw_content_parts = []
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    raw_content_parts.append(piece)
            raw_content = "".join(raw_content_parts)
            
            sample = clean_and_extract_json(raw_content)
            
            # 2. 自動格式與 Schema 檢驗與容錯轉換
            messages = sample.get("messages", [])
            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        assistant_text = content.strip()
                        if "<tool_call>" not in assistant_text and assistant_text.startswith("{"):
                            try:
                                json.loads(assistant_text)
                                assistant_text = f"<tool_call>{assistant_text}</tool_call>"
                                msg["content"] = assistant_text
                            except Exception:
                                pass
                    elif isinstance(content, dict):
                        assistant_text = f"<tool_call>{json.dumps(content, ensure_ascii=False)}</tool_call>"
                        msg["content"] = assistant_text
                    elif "tool_calls" in msg and msg["tool_calls"]:
                        tc_obj = msg["tool_calls"]
                        if isinstance(tc_obj, list) and len(tc_obj) > 0:
                            first_tc = tc_obj[0]
                            if isinstance(first_tc, dict) and "function" in first_tc:
                                first_tc = first_tc["function"]
                            assistant_text = f"<tool_call>{json.dumps(first_tc, ensure_ascii=False)}</tool_call>"
                            msg["content"] = assistant_text
                    elif isinstance(msg, dict) and ("name" in msg or "args" in msg):
                        assistant_text = f"<tool_call>{json.dumps(msg, ensure_ascii=False)}</tool_call>"
                        msg["content"] = assistant_text
            
            if not assistant_text:
                sample_preview = json.dumps(sample, ensure_ascii=False)[:300]
                raise ValueError(f"產生的 JSON 樣本中 messages 缺少 assistant 的 content 內容。模型實際輸出的結構片段為:\n{sample_preview}")
            
            # 測試提取與校驗
            try:
                args = extract_tool_call(assistant_text)
            except ValueError as ve:
                raise ValueError(f"提取 <tool_call> 標籤失敗或內部 JSON 損毀: {str(ve)}")
                
            is_valid, err_msg = validate_record(args, sub_categories, sub_accounts)
            
            if is_valid:
                if "date" not in args:
                    is_valid = False
                    err_msg = "缺少 date 欄位，請在 tool call 中務必填入 date 欄位並依今日基準計算"
                elif not re.match(r"^\d{4}-\d{2}-\d{2}$", str(args["date"])):
                    is_valid = False
                    err_msg = f"date 格式錯誤: {args['date']}，必須為 YYYY-MM-DD"

            # 3.5 檢查分類與類型語意一致性
            if is_valid:
                cat = args.get("category", "")
                rtype = args.get("type", "")
                if rtype == "income" and cat not in local_income_cats:
                    is_valid = False
                    err_msg = f"分類 '{cat}' 不適用於 type=income"
                elif rtype == "expense" and cat not in local_expense_cats:
                    is_valid = False
                    err_msg = f"分類 '{cat}' 不適用於 type=expense"

            # 4. 檢查內容是否與歷史已生成語料重複
            if is_valid and recent_history:
                user_text = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_text = msg.get("content", "").strip()
                desc_text = str(args.get("description", "")).strip()
                
                if user_text and (user_text in recent_history or (desc_text and desc_text in recent_history)):
                    is_valid = False
                    err_msg = f"生成內容/備註與歷史語料重複 ('{user_text}')，請重新思考全新的事由與表達方式"

            if is_valid:
                # 由 Python 自動將完整的 tool_definition 縫合至 system content 中，確保 Token 大幅節省且格式 100% 精準
                full_system_content = f"今天是 {ref_date_str}。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(tool_definition, ensure_ascii=False)}"
                for msg in messages:
                    if msg.get("role") == "system":
                        msg["content"] = full_system_content
                return sample
            else:
                raise ValueError(f"欄位與 Schema 限制驗證不合規: {err_msg}")
                
        except ValueError as ve:
            error_detail = str(ve)
            try:
                print(f"警告 (修復嘗試 {attempt+1}/{max_retries}): {error_detail}")
            except UnicodeEncodeError:
                print(f"Warning (retry {attempt+1}/{max_retries}): encoding error")
            if attempt < max_retries - 1:
                # 重新重置提示詞，防止壞掉的 context 污染後續生成
                sep = chr(10) * 2
                fix_msg = f"【重要修復提醒】上一輪生成失敗，原因：{error_detail}{sep}請嚴格輸出合法且可被解析的 JSON 物件。"
                history_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{prompt}{sep}{fix_msg}"}
                ]
            else:
                print(f"錯誤: 已達到最大自我修復重試次數 ({max_retries})，放棄此輪樣本生成。")
        except RateLimitError as e:
            backoff_time = random.uniform(5.0, 10.0)
            print(f"[RateLimitError 429] 請求頻率過快: {e}。隨機退避等待 {backoff_time:.2f} 秒後重試...")
            await asyncio.sleep(backoff_time)
        except APIError as e:
            print(f"[APIError {e.code}] 發生 API 錯誤 ({e.message})。隨機等待 3 秒後重試...")
            await asyncio.sleep(random.uniform(2.0, 4.0))
        except Exception as e:
            try:
                print(f"請求拋出其他異常: {e}。隨機等待 3 秒後重試...")
            except UnicodeEncodeError:
                print(f"Request error: {e}")
            await asyncio.sleep(3.0)
            
    return None

async def generate_dataset(api_url, api_key, model_name, num_samples, out_dir, concurrency=2, rest_interval=0, rest_duration=1, thinking_budget=0, bias_categories=None):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_generated.jsonl")
    
    # 1. 偵測並讀取現有的生成存檔以支援斷點續傳
    all_samples = []
    history_user_texts = []
    existing_counts = {"Level-1 (Simple)": 0, "Level-2 (Noise)": 0, "Level-3 (Reasoning)": 0}
    
    if os.path.exists(raw_path):
        print(f"偵測到歷史生成的資料存檔 {raw_path}，正在載入進度以支援斷點續傳...")
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                    all_samples.append(s)
                    for msg in s.get("messages", []):
                        if msg.get("role") == "user":
                            history_user_texts.append(msg["content"].strip())
                    diff = s.get("difficulty_level")
                    if diff:
                        # 容錯比對：Level-1 與 Level-1 (Simple) 視為一致
                        if diff.startswith("Level-1"):
                            diff_key = "Level-1 (Simple)"
                        elif diff.startswith("Level-2"):
                            diff_key = "Level-2 (Noise)"
                        elif diff.startswith("Level-3"):
                            diff_key = "Level-3 (Reasoning)"
                        else:
                            diff_key = diff
                        
                        if diff_key in existing_counts:
                            existing_counts[diff_key] += 1
                except Exception:
                    pass
        print(f"歷史已載入進度：{existing_counts} (已累積 {len(history_user_texts)} 筆歷史語料特徵用於排重)")
    
    # 計算各難度目標比例 (4:3:3)
    num_l1 = int(num_samples * 0.4)
    num_l2 = int(num_samples * 0.3)
    num_l3 = num_samples - num_l1 - num_l2
    
    tasks_pool = [
        ("Level-1 (Simple)", num_l1),
        ("Level-2 (Noise)", num_l2),
        ("Level-3 (Reasoning)", num_l3)
    ]

    # 初始化 AsyncOpenAI 客戶端，並設定自訂 Limits 防止 httpx 默認連接池限制並行數
    import httpx
    limits = httpx.Limits(max_connections=concurrency * 10, max_keepalive_connections=concurrency * 2)
    client = AsyncOpenAI(
        api_key=api_key, 
        base_url=api_url,
        http_client=httpx.AsyncClient(limits=limits)
    )
    
    import time
    last_rest_time = time.time()
    
    for difficulty, count in tasks_pool:
        completed = existing_counts.get(difficulty, 0)
        if completed >= count:
            print(f"[{difficulty}] 歷史已生成 {completed}/{count} 筆，直接跳過。")
            continue
            
        print(f"開始生成 {difficulty} 數據，目標數量: {count} 筆 (當前進度: {completed}/{count})...")
        
        # 循環直到生成足夠的合規數據，使用定長批次控制避免 race condition
        while completed < count:
            # 顯卡防過熱保護休息邏輯
            if rest_interval > 0:
                elapsed_min = (time.time() - last_rest_time) / 60.0
                if elapsed_min >= rest_interval:
                    print(f"\n[顯卡保護] 已持續執行 {elapsed_min:.2f} 分鐘 (設定閾值: {rest_interval} 分鐘)")
                    print(f"現在開始暫停休息 {rest_duration} 分鐘，以防顯卡過熱...")
                    await asyncio.sleep(rest_duration * 60)
                    last_rest_time = time.time()
                    print("[顯卡保護] 休息結束，恢復生成！\n")

            needed = count - completed
            batch_size = min(concurrency, needed)
            print(f"[{difficulty}] 當前批次發送 {batch_size} 筆並行請求 (剩餘目標: {needed})...")
            
            async def worker():
                nonlocal completed
                if completed >= count:
                    return
                # 每個 task 發起前增加微小的時間抖動，避免同時打擊 API
                await asyncio.sleep(random.uniform(0.1, 0.5))
                ref_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 364))
                
                sample = await generate_single_sample(
                    client, model_name, difficulty, CATEGORIES, ACCOUNTS, ref_date,
                    recent_history=history_user_texts, thinking_budget=thinking_budget,
                    bias_categories=bias_categories
                )
                if sample:
                    all_samples.append(sample)
                    for msg in sample.get("messages", []):
                        if msg.get("role") == "user":
                            history_user_texts.append(msg["content"].strip())
                    # 實時追加寫入 raw 存檔
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    completed += 1
                    if completed % 5 == 0 or completed == count:
                        print(f"[{difficulty}] 已完成 {completed}/{count} 筆")
                        
            tasks = [worker() for _ in range(batch_size)]
            await asyncio.gather(*tasks)

    # 隨機打亂資料集
    random.shuffle(all_samples)
    
    # 劃分訓練集與測試集 (80% 訓練, 20% 測試)
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    # 寫入檔案
    train_path = os.path.join(out_dir, "train_strata.jsonl")
    test_path = os.path.join(out_dir, "test_strata.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\n資料集生成完畢！")
    print(f"  - 訓練集路徑: {train_path} ({len(train_data)} 筆)")
    print(f"  - 測試集路徑: {test_path} ({len(test_data)} 筆)")

def main():
    parser = argparse.ArgumentParser(description="使用 OpenAI SDK 批量生成分層記帳 dataset 腳本")
    parser.add_argument("--api_url", type=str, default=os.environ.get("OPENAI_BASE_URL"), help="您的 API 端點 (可設於 .env 的 OPENAI_BASE_URL)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="API 金鑰 (可設於 .env 的 OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-mtp"), help="用於生成數據集的大模型名稱 (可設於 .env 的 LLM_MODEL)")
    parser.add_argument("--count", type=int, default=100, help="總共生成的樣本數量")
    parser.add_argument("--out_dir", type=str, default="./dataset", help="資料集輸出目錄")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", 2)), help="並行生成的呼叫數量 (預設為 2，可設於 .env 的 CONCURRENCY)")
    parser.add_argument("--rest_interval", type=float, default=float(os.environ.get("REST_INTERVAL", 0)), help="每隔多少分鐘進行一次休息 (預設 0 = 不休息，可設於 .env 的 REST_INTERVAL)")
    parser.add_argument("--rest_duration", type=float, default=float(os.environ.get("REST_DURATION", 1)), help="每次休息的分鐘數 (預設 1，可設於 .env 的 REST_DURATION)")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Qwen 推理模型的 thinking_budget (0 = 關閉 thinking，預設 0)")
    parser.add_argument("--bias_categories", type=str, default="", help="逗號分隔的類別清單，70% 機率偏向這些類別生成 (預設空 = 不偏)")
    
    args = parser.parse_args()
    
    # 強制安全檢驗，絕不寫死金鑰
    if not args.api_key:
        raise ValueError(
            "\n[安全錯誤] 未提供 API 金鑰！\n"
            "請在 .env 檔案中設定 OPENAI_API_KEY，或設定環境變數 OPENAI_API_KEY，或使用 --api_key 參數傳入金鑰。\n"
            "例如在 .env 中寫入：\n"
            "  OPENAI_API_KEY=your_key_here\n"
        )
        
    if not args.api_url:
        raise ValueError(
            "\n[安全錯誤] 未提供 API 端點！\n"
            "請在 .env 檔案中設定 OPENAI_BASE_URL，或設定環境變數 OPENAI_BASE_URL，或使用 --api_url 參數傳入端點。\n"
            "例如在 .env 中寫入：\n"
            "  OPENAI_BASE_URL=https://api.openai.com/v1\n"
        )
    
    asyncio.run(generate_dataset(
        args.api_url, 
        args.api_key, 
        args.model, 
        args.count, 
        args.out_dir,
        args.concurrency,
        args.rest_interval,
        args.rest_duration,
        args.thinking_budget,
        [c.strip() for c in args.bias_categories.split(",") if c.strip()] if args.bias_categories else None
    ))

if __name__ == "__main__":
    main()
