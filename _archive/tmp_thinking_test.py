import asyncio, json, os, random, re, sys
from datetime import date
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")
import generate_dataset as gd

ref = "2026-07-20"
system_prompt = gd.GENERATOR_SYSTEM_PROMPT.replace("{ref_date}", ref).replace("{{", "{").replace("}}", "}")
newline = chr(10)

VARIANTS = {
    "thinking-off":        {"chat_template_kwargs": {"enable_thinking": False}},
    "budget-512":          {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}},
    "budget-1024":         {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024}},
    "budget-2048":         {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 2048}},
}

def build_prompt():
    sub_categories = list(gd.CATEGORIES)
    extra_cats = random.sample(gd.EXTRA_CATEGORIES_POOL, k=random.randint(1, 2))
    local_exp = set(gd.EXPENSE_CATS); local_inc = set(gd.INCOME_CATS)
    for ec_name, ec_type in extra_cats:
        sub_categories.append(ec_name)
        (local_exp if ec_type == "expense" else local_inc).add(ec_name)
    sub_accounts = ["現金"] + random.sample([a for a in gd.ACCOUNTS if a != "現金"], k=random.randint(3, 5))
    target_category = random.choice(sub_categories)
    target_account = random.choice(sub_accounts)
    is_income = target_category in local_inc
    expected_type = "income" if is_income else "expense"
    scenario = random.choice(gd.SCENARIOS)
    lang_style = random.choice(gd.LANG_STYLES)
    return (
        f"請生成 1 筆包含 <tool_call> 的對話樣本。{newline}"
        f"難度等級: Level-1 (Simple){newline}"
        f"可用的分類清單: {sub_categories}{newline}"
        f"可用的帳戶清單: {sub_accounts}{newline}"
        f"參考日期 (今天): {ref}{newline}"
        f"請確保對話涉及的分類為 '{target_category}'，帳戶為 '{target_account}'。{newline}"
        f"請注意，分類 '{target_category}' 應使用 type='{expected_type}'，切勿混用。{newline}"
        f"【多樣性要求】{newline}"
        f"1. 必須圍繞以下情境展開故事：{scenario}{newline}"
        f"2. 口語風格限制：{lang_style}{newline}{newline}"
        f"【輸出格式要求】{newline}"
        f"- 請輸出一個 JSON 物件，包含 difficulty_level 和 messages 兩個欄位。{newline}"
        f"- 請勿將 JSON 包裝在陣列中。{newline}"
        f"- assistant 的 content 欄位請輸出「純 JSON 字串」，不要包含 <tool_call>、</tool_call> 或任何 XML 標籤。{newline}"
        f"- 使用 'args' 而非 'arguments'。禁止 emoji。只能使用清單中的分類與帳戶。{newline}"
    )

async def run_one(client, extra_body, prompt):
    t0 = asyncio.get_event_loop().time()
    try:
        resp = await client.chat.completions.create(
            model="vllm/Qwen3.6-27B",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=2048, stream=True,
            extra_body=extra_body,
        )
        content_parts = []
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
        elapsed = asyncio.get_event_loop().time() - t0
        content = "".join(content_parts)
        try:
            sample = gd.clean_and_extract_json(content)
            msgs = sample.get("messages", [])
            ass = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            tc = re.search(r"<tool_call>(.*?)</tool_call>", ass)
            has_tc = bool(tc)
            obj_ok = False
            if tc:
                try:
                    args = json.loads(tc.group(1)).get("args", {})
                    obj_ok = bool(args.get("category") and args.get("description"))
                except Exception:
                    pass
            else:
                try:
                    inner = json.loads(ass)
                    args = inner.get("args", inner)
                    obj_ok = bool(args.get("category") and args.get("description") and args.get("date"))
                except Exception:
                    pass
            return {"elapsed": round(elapsed, 1), "len": len(content), "has_tc": has_tc,
                    "obj_ok": obj_ok, "error": None}
        except Exception as e:
            return {"elapsed": round(elapsed, 1), "len": len(content), "has_tc": False,
                    "obj_ok": False, "error": str(e)[:120]}
    except Exception as e:
        return {"elapsed": 0, "len": 0, "has_tc": False, "obj_ok": False, "error": f"API {type(e).__name__}: {str(e)[:80]}"}

async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
    n = 3
    for name, extra in VARIANTS.items():
        print(f"\n=== {name} (n={n}) ===")
        results = []
        for i in range(n):
            results.append(await run_one(client, extra, build_prompt()))
            await asyncio.sleep(1.5)
        ok = sum(1 for r in results if r["obj_ok"])
        apis = [r for r in results if r["error"] and r["error"].startswith("API")]
        times = [r["elapsed"] for r in results if r["elapsed"]]
        print(f"  obj_ok: {ok}/{n}, api_errors: {len(apis)}, avg_time: {sum(times)/len(times) if times else 0:.1f}s")
        for r in results:
            tag = "OK" if r["obj_ok"] else ("API-ERR" if r["error"] and r["error"].startswith("API") else "FAIL")
            print(f"    {tag} {r['elapsed']}s len={r['len']} has_tc={r['has_tc']} {r['error'] or ''}")

asyncio.run(main())
