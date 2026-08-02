import os, json, sys, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, ".")
from generate_dataset import GENERATOR_SYSTEM_PROMPT, CATEGORIES, ACCOUNTS, SCENARIOS, LANG_STYLES
from datetime import date
import random

ref_date = date(2026, 7, 20)
ref_date_str = ref_date.isoformat()
system_prompt = GENERATOR_SYSTEM_PROMPT.replace("{ref_date}", ref_date_str).replace("{{", "{").replace("}}", "}")
sub_categories = list(CATEGORIES)
sub_accounts = ["現金"] + random.sample([a for a in ACCOUNTS if a != "現金"], k=random.randint(3, 5))

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
            "date": {"type": "string"}
        },
        "required": ["amount", "category", "account", "type", "date"]
    }
}

newline = chr(10)
target_category = random.choice(sub_categories)
target_account = random.choice(sub_accounts)
scenario = random.choice(SCENARIOS)
lang_style = random.choice(LANG_STYLES)

prompt = (
    f"請輸出一個 JSON 物件，包含 difficulty_level 和 messages 兩個欄位。{newline}"
    f"難度等級: Level-1 (Simple){newline}"
    f"可用的分類清單: {sub_categories}{newline}"
    f"可用的帳戶清單: {sub_accounts}{newline}"
    f"參考日期 (今天): {ref_date_str}{newline}"
    f"請確保對話涉及的分類為 '{target_category}'，帳戶為 '{target_account}'。{newline}"
    f"情境：{scenario}；口語風格：{lang_style}{newline}"
    f"messages 包含 system、user、assistant 三筆物件。{newline}"
    f"assistant 的 content 欄位為一個純 JSON 字串物件（不加任何 XML 標籤，不用 <tool_call>），形如：{newline}"
    f"{{\"name\": \"add_record\", \"args\": {{\"amount\": 150, \"category\": \"{target_category}\", \"account\": \"{target_account}\", \"description\": \"消費說明\", \"type\": \"expense\", \"date\": \"{ref_date_str}\"}}}}{newline}"
    f"- 使用 'args' 而非 'arguments'。禁止 emoji。只能使用清單中的分類與帳戶。{newline}"
)

async def main():
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    for extra in [
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"thinking": {"type": "disabled"}},
        {},
    ]:
        tag = list(extra.keys())[0] if extra else "no-thinking-flag"
        print(f"\n=== extra_body={tag} ===")
        t0 = asyncio.get_event_loop().time()
        try:
            resp = await client.chat.completions.create(
                model="vllm/Qwen3.6-27B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
                stream=False,
                extra_body=extra if extra else None,
            )
            elapsed = asyncio.get_event_loop().time() - t0
            msg = resp.choices[0].message
            content = msg.content or ""
            print(f"Elapsed: {elapsed:.1f}s, content length: {len(content)}")
            print(f"finish_reason: {resp.choices[0].finish_reason}")
            print(f"usage: {resp.usage}")
            print(f"Full content: {content}")
            print(f"---FULL END---")
        except Exception as e:
            print(f"API error: {type(e).__name__}: {e}")
        break  # only test first flag

asyncio.run(main())
