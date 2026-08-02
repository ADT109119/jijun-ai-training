import asyncio, json, os, random, re, sys
from datetime import date
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")
import generate_dataset as gd
from filter_dataset import is_description_aligned

CATEGORIES = gd.CATEGORIES
ACCOUNTS = gd.ACCOUNTS
SCENARIOS = gd.SCENARIOS
LANG_STYLES = gd.LANG_STYLES
EXTRA_CATEGORIES_POOL = gd.EXTRA_CATEGORIES_POOL
EXPENSE_CATS = set(gd.EXPENSE_CATS)
INCOME_CATS = set(gd.INCOME_CATS)

V0 = gd.GENERATOR_SYSTEM_PROMPT.replace("{ref_date}", "2026-07-20").replace("{{", "{").replace("}}", "}")

V1 = """你是一個專門生成機器學習訓練數據集的 AI 助手。你的任務是批量產生「中文口語記帳與 Tool Call 對話軌跡」的訓練資料。

我們有一個名為 add_record 的記帳工具 (tool)，其參數結構如下:
- amount: 數值，交易金額 (必須 > 0，請用 realistic 的金額：支出 50~5000 元，收入 1000~200000 元)
- category: 字串，交易分類 (必須從給定的分類清單中選擇)
- account: 字串，支付媒介 (必須從給定的帳戶清單中選擇)
- description: 字串，消費的具體說明備註
- type: 字串，必須為 "expense" (支出) 或 "income" (收入)
- date: 字串，交易日期 (必須符合 YYYY-MM-DD 格式，以 2026-07-20 為今日基準計算)

你每次需要根據我指定的「難度等級」和給定的「分類清單、帳戶清單」，產生 1 筆對話資料。
輸出格式必須嚴格為以下 JSON 結構（不要包裝在陣列中，不要加 Markdown 圍欄）：
{
  "difficulty_level": "Level-1 (Simple)",
  "messages": [
    {
      "role": "system",
      "content": "今天是 2026-07-20。你是一個記帳助理。"
    },
    {
      "role": "user",
      "content": "剛剛吃午餐花了 150 元付現"
    },
    {
      "role": "assistant",
      "content": "{\\"name\\": \\"add_record\\", \\"args\\": {\\"amount\\": 150, \\"category\\": \\"飲食\\", \\"account\\": \\"現金\\", \\"description\\": \\"午餐\\", \\"type\\": \\"expense\\", \\"date\\": \\"2026-07-20\\"}}"
    }
  ]
}

請確保：
1. user 的內容必須極為日常、口語化，符合台灣繁體中文的用語習慣。
2. 根據不同的難度等級設計對話：
   - Level-1 (Simple)：直白簡單，無雜訊。
   - Level-2 (Noise)：加入日常閒聊與修飾字詞，但金額與分類仍明確。
   - Level-3 (Reasoning)：需要簡單推理或時間計算（例：「昨天買的三本小說今天送到了」，模型需推算日期為昨天，且將小說歸類為教育或娛樂）。
3. 使用者的日常口語記帳輸入如果有提及日期、相對時間，你必須以系統設定的參考日期 2026-07-20 為基準點計算出正確的 date 欄位值。如果使用者沒有提及任何時間，則將 date 設為 '2026-07-20'（今天）。
4. 你必須嚴格遵循使用者訊息中指定的 target_category 與 target_account，輸出的 tool_call 中 category 與 account 欄位必須與之完全相同，不得更改。同時，故事描述必須與 target_category 的語意一致。
5. 請使用 "args" 作為 key，不要使用 "arguments"。
6. 禁止使用任何 emoji 或顏文字（如 ☁️、😊、QQ 等），僅使用純文字。
"""

V2 = V1 + """

【額外強制要求】
- 使用者的口語描述文字中，必須直接出現與 target_category 相關的具體關鍵詞（例如 target_category=飲食，描述就要提到便當、餐廳、小吃、飲料等；target_category=交通，就要提到公車、捷運、計程車、油錢等）。描述必須清楚對應 target_category 的語意。
"""

async def generate_one(client, model, sys_prompt, n=30, concurrency=4):
    results = []
    sem = asyncio.Semaphore(concurrency)
    newline = chr(10)
    async def worker(_):
        sub_categories = list(CATEGORIES)
        extra_cats = random.sample(EXTRA_CATEGORIES_POOL, k=random.randint(1, 2))
        local_exp = set(EXPENSE_CATS); local_inc = set(INCOME_CATS)
        for ec_name, ec_type in extra_cats:
            sub_categories.append(ec_name)
            (local_exp if ec_type == "expense" else local_inc).add(ec_name)
        sub_accounts = ["現金"] + random.sample([a for a in ACCOUNTS if a != "現金"], k=random.randint(3, 5))
        target_category = random.choice(sub_categories)
        target_account = random.choice(sub_accounts)
        is_income = target_category in local_inc
        expected_type = "income" if is_income else "expense"
        scenario = random.choice(SCENARIOS)
        lang_style = random.choice(LANG_STYLES)
        ref = "2026-07-20"
        prompt = (
            f"請生成 1 筆包含 <tool_call> 的對話樣本。{newline}"
            f"難度等級: Level-1 (Simple){newline}"
            f"可用的分類清單: {sub_categories}{newline}"
            f"可用的帳戶清單: {sub_accounts}{newline}"
            f"參考日期 (今天): {ref}{newline}"
            f"target_category: '{target_category}'，target_account: '{target_account}'。你必須使用這兩個值，不得更改。{newline}"
            f"請注意，分類 '{target_category}' 應使用 type='{expected_type}'，切勿混用。{newline}"
            f"【多樣性要求】{newline}"
            f"1. 必須圍繞以下情境展開故事：{scenario}{newline}"
            f"2. 口語風格限制：{lang_style}{newline}{newline}"
            f"【輸出格式要求】{newline}"
            f"- 請輸出一個 JSON 物件，包含 difficulty_level 和 messages 兩個欄位。{newline}"
            f"- 請勿將 JSON 包裝在陣列中。{newline}"
            f"- messages 是一個陣列，包含 system、user、assistant 三筆物件。{newline}"
            f"- assistant 的 content 欄位請輸出「純 JSON 字串」，不要包含 <tool_call>、</tool_call> 或任何 XML 標籤。{newline}"
            f"- 使用 'args' 而非 'arguments'。禁止 emoji。只能使用清單中的分類與帳戶。{newline}"
        )
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1024, stream=False,
                    )
                    raw = resp.choices[0].message.content or ""
                    sample = gd.clean_and_extract_json(raw)
                    msgs = sample.get("messages", [])
                    ass = ""
                    for m in msgs:
                        if m.get("role") == "assistant":
                            c = m.get("content")
                            if isinstance(c, str):
                                ass = c.strip()
                            elif isinstance(c, dict):
                                ass = json.dumps(c, ensure_ascii=False)
                    tc = re.search(r"<tool_call>(.*?)</tool_call>", ass)
                    if tc:
                        obj = json.loads(tc.group(1))
                    else:
                        try:
                            obj = json.loads(ass)
                        except Exception:
                            continue
                    args = obj.get("args", obj)
                    cat = args.get("category", "")
                    desc = args.get("description", "")
                    aligned = is_description_aligned(cat, desc)
                    forced = (cat == target_category)
                    results.append({"cat": cat, "desc": desc, "aligned": aligned, "forced": forced,
                                    "target": target_category})
                    return
                except Exception as e:
                    if "429" in str(e) or "503" in str(e):
                        await asyncio.sleep(random.uniform(3, 7))
                    else:
                        await asyncio.sleep(1)
                    continue
            results.append({"cat": "ERROR", "desc": "fail", "aligned": False, "forced": False, "target": target_category})
    await asyncio.gather(*[worker(i) for i in range(n)])
    return results

async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=45)
    for name, sys_prompt in [("V0-baseline", V0), ("V1-force-cat", V1), ("V2-force+kw", V2)]:
        print(f"[start] {name}", flush=True)
        results = await generate_one(client, "gemma-4-e4b", sys_prompt, n=25, concurrency=3)
        total = len(results)
        aligned = sum(1 for r in results if r["aligned"])
        forced = sum(1 for r in results if r["forced"])
        forced_aligned = sum(1 for r in results if r["forced"] and r["aligned"])
        print(f"\n=== {name} (n={total}) ===")
        print(f"  aligned rate: {aligned}/{total} = {aligned/total*100:.1f}%")
        print(f"  forced (cat==target): {forced}/{total} = {forced/total*100:.1f}%")
        print(f"  forced AND aligned: {forced_aligned}/{total}")
        bad = [(r['target'], r['cat'], r['desc'][:30]) for r in results if not r['aligned']][:5]
        for t, c, d in bad:
            print(f"    target={t} -> cat={c} | {d}")
        print(f"[done] {name}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
