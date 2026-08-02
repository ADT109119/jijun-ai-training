import os, json, random, sys, asyncio, time
from openai import AsyncOpenAI
from batch_generate_v2 import (
    build_prefix_map, build_colloquial_prefix_map,
    REFERENCE_DATES, CATEGORIES, ACCOUNTS, make_sample
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REF_DATE = REFERENCE_DATES[0]
rel_map = build_prefix_map(REF_DATE)
coll_map = build_colloquial_prefix_map(REF_DATE)
ALL_DATE_MAP = {**rel_map, **coll_map}

async def generate_one(client, model, expr, cat, acc, amt, rtype):
    d = ALL_DATE_MAP[expr]
    rtype_cn = "收入" if rtype == "income" else "支出"
    prompt = f"Say in one sentence: {expr} {cat} {amt}元 {acc} {rtype_cn}"
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            # Try content first
            txt = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            # Fallback: try reasoning field
            if not txt:
                r = resp.choices[0].message.reasoning or ""
                # Extract last substantive line from reasoning
                for line in reversed(r.split("\n")):
                    ls = line.strip()
                    if ls and len(ls) > 5 and len(ls) < 200 and not ls.startswith("Here") and not ls.startswith("-"):
                        txt = ls
                        break
            if len(txt) < 5 or len(txt) > 200:
                raise ValueError(f"bad len {len(txt)}")
            return txt, d
        except Exception as e:
            if attempt == 2:
                return None, None
            await asyncio.sleep(0.3)

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    api_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("LLM_MODEL", "vllm/Qwen3.6-27B")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    concurrency = int(os.environ.get("CONCURRENCY", 6))

    client = AsyncOpenAI(api_key=api_key, base_url=api_url, timeout=120)
    out_path = "dataset/llm_generated.jsonl"

    existing = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing += 1
    print(f"Existing: {existing}, Target: {num_samples} more ({existing+num_samples} total)")

    used_keys = set()
    sem = asyncio.Semaphore(concurrency)
    start = time.time()
    done = existing
    total = existing + num_samples

    async def worker(wid):
        nonlocal done
        while done < total:
            async with sem:
                if done >= total:
                    break
                # Pick expression: bias toward colloquial (70%) vs relative (30%)
                if random.random() < 0.30:
                    expr = random.choice(list(rel_map.keys()))
                else:
                    expr = random.choice(list(coll_map.keys()))
                cat = random.choice(CATEGORIES)
                acc = random.choice(ACCOUNTS)
                rtype = "expense" if random.random() < 0.6 else "income"
                amt = random.randint(15, 8000) * 5 if rtype == "expense" else random.randint(1000, 80000) * 10
                key = (expr, cat, acc, amt, rtype)
                if key in used_keys:
                    continue
                used_keys.add(key)
                result = await generate_one(client, model_name, expr, cat, acc, amt, rtype)
                if result is None or result[0] is None:
                    continue
                text, d = result
                sample = make_sample("Level-2 (Noise)", cat, acc, amt, rtype, text, d, REF_DATE)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                done += 1
                elapsed = time.time() - start
                if done % 25 == 0:
                    rate = (done - existing) / elapsed * 60
                    print(f"  {done-existing}/{num_samples} ({rate:.0f}/min)", flush=True)

    workers = [worker(i) for i in range(concurrency)]
    await asyncio.gather(*workers)
    elapsed = time.time() - start
    n = done - existing
    print(f"\nDone! {n} samples in {elapsed:.0f}s ({n/elapsed*60:.1f}/min)")

if __name__ == "__main__":
    asyncio.run(main())
