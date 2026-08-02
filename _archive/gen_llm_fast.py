import sys, os, json, asyncio, random, httpx
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import AsyncOpenAI
from generate_dataset import generate_single_sample, CATEGORIES, ACCOUNTS
from dotenv import load_dotenv
load_dotenv()

model = 'qwen3.5-9b-ultra-uncensored-heretic-v2'
output = './dataset/llm_generated.jsonl'

client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'])

counts = {'Level-1 (Simple)': 0, 'Level-2 (Noise)': 0, 'Level-3 (Reasoning)': 0}
if os.path.exists(output):
    with open(output, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                for k in counts:
                    if s.get('difficulty_level','').startswith(k[:7]):
                        counts[k] += 1
                        break
print(f'Existing: {sum(counts.values())} -> {counts}', flush=True)

TARGET_PER_LEVEL = 100
total_existing = sum(counts.values())
print(f'Need {TARGET_PER_LEVEL * 3 - total_existing} more samples', flush=True)

async def worker(diff, needed):
    for i in range(needed):
        await asyncio.sleep(random.uniform(0.2, 0.5))
        sample = await generate_single_sample(client, model, diff, CATEGORIES, ACCOUNTS)
        if sample:
            counts[diff] += 1
            with open(output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            if i % 5 == 0 or i == needed - 1:
                print(f'  [{diff}] {counts[diff]}/{TARGET_PER_LEVEL}', flush=True)

async def main():
    tasks = []
    for diff, target in [('Level-1 (Simple)', TARGET_PER_LEVEL),
                          ('Level-2 (Noise)', TARGET_PER_LEVEL),
                          ('Level-3 (Reasoning)', TARGET_PER_LEVEL)]:
        needed = target - counts[diff]
        if needed > 0:
            tasks.append(worker(diff, needed))
    if tasks:
        await asyncio.gather(*tasks)
    print(f'Done! Total: {sum(counts.values())} -> {counts}', flush=True)

asyncio.run(main())
