import os
import json
import random
import asyncio
import argparse
from openai import AsyncOpenAI, APIError, RateLimitError
from trainer.validate_json import extract_tool_call, validate_record

# 擴充更豐富的預設分類與帳戶，增加多樣性
CATEGORIES = [
    "餐飲飲食", "休閒娛樂", "交通出行", "生活繳費", "醫療保健", 
    "教育學習", "日常雜貨", "服飾美妝", "數位服務", "投資理財", 
    "薪資收入", "獎金紅利", "副業外快", "寵物支出", "房租房貸", 
    "保險費用", "人情往來", "家居裝修", "3C電子", "運動健身"
]

ACCOUNTS = [
    "現金", "信用卡", "悠遊卡", "一卡通", "街口支付", 
    "LINE Pay", "Apple Pay", "Google Pay", "郵局帳戶", 
    "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"
]

# 新增隨機情境池，打破 LLM 單一邏輯複製，增強數據分佈多樣性
SCENARIOS = [
    "上班族日常通勤、商業午餐或加班計程車報銷",
    "大學生購買二手教科書、分期付款買筆電，或與社團同學聚餐",
    "家庭主夫/主婦在超市/量販店買菜、購買家庭日用品或繳納水電瓦斯費",
    "自由工作者在咖啡廳辦公點咖啡、購買軟體訂閱，或收到客戶的專案款項",
    "情侶約會去高檔餐廳慶祝、看電影、或節日送禮物",
    "出國旅遊（如日本、韓國）購買藥妝、預訂住宿、機票，或用外幣進行刷卡消費",
    "寵物飼主購買貓砂貓罐頭、帶寵物去獸醫院看診或打預防針",
    "發薪日收到公司薪水、發放年終獎金，或賣掉二手舊手機獲得的二手變賣收入",
    "網購衣服配件、買電子遊戲（如 Steam）、或者訂閱串流服務（如 Netflix, Spotify, YouTube Premium）",
    "身體不舒服去診所掛號看醫生、去藥局買保健食品，或是定期的保險保費扣款",
    "投資理財：買入零股股票、領到股息利息、基金定額扣款，或是虛擬貨幣交易",
    "親友生日包紅包、婚禮人情往來禮金、請朋友喝飲料，或朋友還錢的收入"
]

# 新增口語化風格池，確保對話句式與長度分佈豐富
LANG_STYLES = [
    "極簡短句，甚至省略主詞或動詞（例：『午餐吃麵 120 現金』）",
    "囉唆、帶有許多心情故事與情境碎碎念的長句（例：『今天下大雨真煩，下班忍不住去全聯大買特買零食，不知不覺花了我五百多塊，刷了信用卡，心在痛』）",
    "倒裝句或順序混亂的表達（例：『刷了 LINE Pay 買星巴克，花了 160 塊今天早上』）",
    "包含數字或貨幣口語的寫法（例：『去屈臣氏買個洗面乳，花了一張藍色小朋友，找回的零錢用悠遊卡嗶了』）",
    "台灣在地日常用語與語助詞（例：『哇賽，剛剛去美聯社買牛奶，用悠遊卡扣了 90 塊耶』）",
    "帶有明確日期指代（例：『上禮拜三去健身房扣款 1200，刷了台新卡』或『昨天領了外包薪水 3 萬存入銀行』）",
    "包含部分英文術語或簡寫的夾雜口語（例：『買了 Udemy 課程花了 50 USD，刷 Visa 卡』）"
]

# 系統 Prompt 與 Tools 定義，用於教導大模型如何生成樣本
# 我們特別在結尾強調了 "JSON" 這個字，以相容各類大模型強制 JSON 輸出的約束
GENERATOR_SYSTEM_PROMPT = """
你是一個專門生成機器學習訓練數據集的 AI 助手。你的任務是批量產生「中文口語記帳與 Tool Call 對話軌跡」的訓練資料。

我們有一個名為 add_record 的記帳工具 (tool)，其參數結構如下:
- amount: 數值，交易金額 (必須 > 0)
- category: 字串，交易分類 (必須從給定的分類清單中選擇)
- account: 字串，支付媒介 (必須從給定的帳戶清單中選擇)
- description: 字串，消費的具體說明備註
- type: 字串，必須為 "expense" (支出) 或 "income" (收入)

你每次需要根據我指定的「難度等級」和給定的「分類清單、帳戶清單」，產生 1 筆對話資料。
你必須嚴格輸出以下符合 JSON 規範的格式（不要包含額外的 Markdown 或說明文字）：
{
  "difficulty_level": "指定的難度等級",
  "messages": [
    {
      "role": "system",
      "content": "你是一個記帳助理。你被賦予了以下 tools:\\n{\\"name\\": \\"add_record\\", \\"description\\": \\"新增記帳記錄\\", \\"parameters\\": { ... }}"
    },
    {
      "role": "user",
      "content": "使用者的日常口語記帳輸入"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\\"name\\": \\"add_record\\", \\"args\\": { ... }}</tool_call>"
    }
  ]
}

請確保：
1. user 的內容必須極為日常、口語化，符合台灣繁體中文的用語習慣。
2. 根據不同的難度等級設計對話：
   - Level-1 (Simple)：直白簡單，無雜訊（例：「剛剛吃午餐花了 150 元付現」）。
   - Level-2 (Noise)：加入日常閒聊與修飾字詞，但金額與分類仍明確（例：「今天天氣真好，下午和朋友喝下午茶花了320元，刷了信用卡，真開心」）。
   - Level-3 (Reasoning)：需要簡單推理或時間計算，或多個項目的綜合（例：「昨天買的三本小說今天送到了，總共花了890元用悠遊卡付的」，模型需推算日期為昨天，且將小說歸類為教育或娛樂）。
3. 輸出的 assistant 欄位中，JSON 的 key 與 value 必須完全合規，且能被 JSON 解析器正確 parse。
"""

async def generate_single_sample(client, model_name, difficulty, categories, accounts):
    """
    發送非同步請求至 NexusLLM/OpenAI API 生成單筆樣本，包含自動自我修正 (Self-Correction) 迴圈
    """
    sub_categories = random.sample(categories, k=random.randint(4, 7))
    sub_accounts = random.sample(accounts, k=random.randint(3, 5))
    
    target_category = random.choice(sub_categories)
    target_account = random.choice(sub_accounts)

    tool_definition = {
        "name": "add_record",
        "description": "新增一筆記帳記錄",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": sub_categories},
                "account": {"type": "string", "enum": sub_accounts},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": ["expense", "income"]}
            },
            "required": ["amount", "category", "account", "type"]
        }
    }

    scenario = random.choice(SCENARIOS)
    lang_style = random.choice(LANG_STYLES)

    prompt = (
        f"請生成 1 筆符合格式的對話樣本。\n"
        f"難度等級: {difficulty}\n"
        f"可用的分類清單: {sub_categories}\n"
        f"可用的帳戶清單: {sub_accounts}\n"
        f"請確保對話涉及的分類為 '{target_category}'，帳戶為 '{target_account}'。\n"
        f"【核心多樣性要求】\n"
        f"1. 必須圍繞以下情境展開故事：{scenario}\n"
        f"2. 口語風格限制：{lang_style}\n\n"
        f"對話中的 system role 的 content 必須嵌入以下 tool 定義：\n"
        f"{json.dumps(tool_definition, ensure_ascii=False)}"
    )

    max_retries = 3
    history_messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        raw_content = ""
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=history_messages,
                temperature=0.7,
                max_tokens=4096,
                stream=True,  # 啟用 Streaming 串流輸出，穩定連接防止 Cloudflare 超時
                extra_body={
                    "reasoning": {"effort": "medium"},
                    "reasoning_effort": "medium",
                    "thinking": {
                        "type": "adaptive",
                        "budget_tokens": 4096
                    },
                    "include_reasoning": True,
                    "include_thoughts": True,
                    "thinking_config": {
                        "thinking_level": "MEDIUM"
                    }
                }
            )
            
            raw_content_list = []
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content is not None:
                        raw_content_list.append(delta.content)
                        
            raw_content = "".join(raw_content_list)
            
            # 1. 嘗試解析 JSON
            try:
                sample = json.loads(raw_content)
            except json.JSONDecodeError as je:
                raise ValueError(f"JSON 格式損毀或含有 Markdown 標記，無法解析: {str(je)}")
            
            # 2. 自動格式與 Schema 檢驗
            messages = sample.get("messages", [])
            assistant_text = ""
            for msg in messages:
                if msg["role"] == "assistant":
                    assistant_text = msg["content"]
            
            if not assistant_text:
                raise ValueError("產生的 JSON 樣本中 messages 缺少 assistant 的 content 內容。")
            
            # 測試提取與校驗
            try:
                args = extract_tool_call(assistant_text)
            except ValueError as ve:
                raise ValueError(f"提取 <tool_call> 標籤失敗或內部 JSON 損毀: {str(ve)}")
                
            is_valid, err_msg = validate_record(args, sub_categories, sub_accounts)
            
            if is_valid:
                return sample
            else:
                raise ValueError(f"欄位與 Schema 限制驗證不合規: {err_msg}")
                
        except ValueError as ve:
            error_detail = str(ve)
            print(f"警告 (修復嘗試 {attempt+1}/{max_retries}): {error_detail}")
            if attempt < max_retries - 1:
                # 將錯誤訊息與失敗的輸出回饋給 LLM，要求它修正
                history_messages.append({"role": "assistant", "content": raw_content})
                feedback_prompt = (
                    f"【修正指令】您剛才輸出的 JSON 格式驗證失敗！原因如下：\n"
                    f"❌ 錯誤細節: {error_detail}\n\n"
                    f"請詳細閱讀上述錯誤，重新輸出一個「修正後」的完整 JSON 樣本。嚴格確保其符合 JSON 格式，"
                    f"且 category 必須為 {sub_categories} 之一，account 必須為 {sub_accounts} 之一，"
                    f"且 assistant 的 content 格式應為 <tool_call>{{\"name\": \"add_record\", \"args\": {{...}}}}</tool_call>。"
                )
                history_messages.append({"role": "user", "content": feedback_prompt})
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
            print(f"請求拋出其他異常: {e}。隨機等待 3 秒後重試...")
            await asyncio.sleep(3.0)
            
    return None

async def generate_dataset(api_url, api_key, model_name, num_samples, out_dir, concurrency=2):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_generated.jsonl")
    
    # 1. 偵測並讀取現有的生成存檔以支援斷點續傳
    all_samples = []
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
        print(f"歷史已載入進度：{existing_counts}")
    
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
    
    for difficulty, count in tasks_pool:
        completed = existing_counts.get(difficulty, 0)
        if completed >= count:
            print(f"[{difficulty}] 歷史已生成 {completed}/{count} 筆，直接跳過。")
            continue
            
        print(f"開始生成 {difficulty} 數據，目標數量: {count} 筆 (當前進度: {completed}/{count})...")
        
        # 循環直到生成足夠的合規數據，使用定長批次控制避免 race condition
        while completed < count:
            needed = count - completed
            batch_size = min(concurrency, needed)
            print(f"[{difficulty}] 當前批次發送 {batch_size} 筆並行請求 (剩餘目標: {needed})...")
            
            async def worker():
                nonlocal completed
                if completed >= count:
                    return
                # 每個 task 發起前增加微小的時間抖動，避免同時打擊 API
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                sample = await generate_single_sample(
                    client, model_name, difficulty, CATEGORIES, ACCOUNTS
                )
                if sample:
                    all_samples.append(sample)
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
    parser.add_argument("--api_url", type=str, default=os.environ.get("OPENAI_BASE_URL"), help="您的 API 端點 (預設讀取環境變數 OPENAI_BASE_URL)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="API 金鑰 (預設讀取環境變數 OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="qwen3.6-35b-a3b-mtp", help="用於生成數據集的大模型名稱")
    parser.add_argument("--count", type=int, default=100, help="總共生成的樣本數量")
    parser.add_argument("--out_dir", type=str, default="./dataset", help="資料集輸出目錄")
    parser.add_argument("--concurrency", type=int, default=2, help="並行生成的呼叫數量 (預設為 2)")
    
    args = parser.parse_args()
    
    # 強制安全檢驗，絕不寫死金鑰
    if not args.api_key:
        raise ValueError(
            "\n[安全錯誤] 未提供 API 金鑰！\n"
            "請設定環境變數 OPENAI_API_KEY 或使用 --api_key 參數傳入金鑰。\n"
            "例如：\n"
            "  Windows PowerShell: $env:OPENAI_API_KEY=\"your_key\"\n"
            "  Linux/macOS: export OPENAI_API_KEY=\"your_key\"\n"
        )
        
    if not args.api_url:
        raise ValueError(
            "\n[安全錯誤] 未提供 API 端點！\n"
            "請設定環境變數 OPENAI_BASE_URL 或使用 --api_url 參數傳入端點。\n"
            "例如：\n"
            "  Windows PowerShell: $env:OPENAI_BASE_URL=\"https://api.openai.com/v1\"\n"
            "  Linux/macOS: export OPENAI_BASE_URL=\"https://api.openai.com/v1\"\n"
        )
    
    asyncio.run(generate_dataset(
        args.api_url, 
        args.api_key, 
        args.model, 
        args.count, 
        args.out_dir,
        args.concurrency
    ))

if __name__ == "__main__":
    main()
