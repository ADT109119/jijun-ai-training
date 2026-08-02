import asyncio, os, sys, time
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")
import generate_dataset as gd

async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), timeout=45)
    sys_prompt = gd.GENERATOR_SYSTEM_PROMPT.replace("{ref_date}", "2026-07-20").replace("{{", "{").replace("}}", "}")
    print(f"=== single sample with fixed-braces system prompt ===", flush=True)
    sub_categories = list(gd.CATEGORIES)
    sub_accounts = ["現金"] + gd.ACCOUNTS[1:5]
    target_category = "飲食"
    target_account = "現金"
    ref = "2026-07-20"
    nl = chr(10)
    prompt = (
        f"請生成 1 筆包含 <tool_call> 的對話樣本。{nl}"
        f"難度等級: Level-1 (Simple){nl}"
        f"可用的分類清單: {sub_categories}{nl}"
        f"可用的帳戶清單: {sub_accounts}{nl}"
        f"參考日期 (今天): {ref}{nl}"
        f"target_category: '{target_category}'，target_account: '{target_account}'。你必須使用這兩個值，不得更改。{nl}"
        f"請注意，分類 '{target_category}' 應使用 type='expense'，切勿混用。{nl}"
        f"【多樣性要求】{nl}1. 必須圍繞以下情境展開故事：上班族中午用餐場景{nl}"
        f"2. 口語風格限制：普通{nl}{nl}"
        f"【輸出格式要求】{nl}"
        f"- 請輸出一個 JSON 物件，包含 difficulty_level 和 messages 兩個欄位。{nl}"
        f"- 請勿將 JSON 包裝在陣列中。{nl}"
        f"- assistant 的 content 欄位請輸出「純 JSON 字串」，不要包含 <tool_call>、</tool_call> 或任何 XML 標籤。{nl}"
        f"- 使用 'args' 而非 'arguments'。禁止 emoji。只能使用清單中的分類與帳戶。{nl}"
    )
    t0 = time.time()
    try:
        r = await asyncio.wait_for(
            client.chat.completions.create(
                model="gemma-4-e4b",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=1024, stream=False,
            ),
            timeout=90,
        )
        print(f"  OK in {time.time()-t0:.1f}s, content len={len(r.choices[0].message.content or '')}", flush=True)
        print(f"  content[:200]: {r.choices[0].message.content[:200]}", flush=True)
    except Exception as e:
        print(f"  FAIL in {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)

asyncio.run(main())
