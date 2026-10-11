"""短語/回歸測試集快速評測：載入 .pt，greedy 生成，逐欄位比對。
用法: python eval_checkpoints.py <ckpt.pt> [dataset.jsonl]
"""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model.model import BookkeepingLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer

ckpt = sys.argv[1]
ds_path = sys.argv[2] if len(sys.argv) > 2 else 'dataset/v4/short_test.jsonl'

tok = BookkeepingTokenizer('jingyaogong/minimind-3')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BookkeepingLM(ModelConfig(vocab_size=tok.vocab_size))
model.load_state_dict(torch.load(ckpt, map_location=device))
model.to(device).eval()

rows = [json.loads(l) for l in open(ds_path, encoding='utf-8') if l.strip()]
FIELDS = ['amount', 'category', 'account', 'description', 'type', 'date']
stats = {f: 0 for f in FIELDS}
n = exact = fmt = 0
t0 = time.time()
fails = []
for r in rows:
    msgs = r['messages']
    prompt = (f"<|im_start|>system\n{msgs[0]['content']}\n<|im_end|>\n"
              f"<|im_start|>user\n{msgs[1]['content']}\n<|im_end|>\n"
              f"<|im_start|>assistant\n")
    input_ids = torch.tensor(tok.encode(prompt, max_length=4096, add_special_tokens=False),
                             dtype=torch.long).unsqueeze(0).to(device)
    gen = []
    cur = input_ids
    with torch.no_grad():
        for _ in range(300):
            logits, _ = model(cur)
            nt = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nt.item())
            if nt.item() in (tok.eos_token_id, tok.encode('<' + '|im_end|>', add_special_tokens=False)[0]):
                break
            cur = torch.cat([cur, nt], dim=-1)
            if cur.shape[1] > 780:
                break
    out = tok.decode(gen)
    n += 1
    brace = out.find('{')
    if brace < 0:
        fails.append((msgs[1]['content'], out[:80], 'FORMAT'))
        continue
    fmt += 1
    try:
        payload, _ = json.JSONDecoder().raw_decode(out[brace:])
        pred = payload.get('args', {})
    except Exception:
        fails.append((msgs[1]['content'], out[:80], 'PARSE'))
        continue
    gold = json.loads(msgs[2]['content'][msgs[2]['content'].index('{'):msgs[2]['content'].rindex('}') + 1])['args']
    allf = True
    for f in FIELDS:
        e, a = gold.get(f), pred.get(f)
        ok = (abs(float(e) - float(a)) < 1e-4) if isinstance(e, (int, float)) and not isinstance(e, bool) and isinstance(a, (int, float)) and not isinstance(a, bool) else (a == e)
        if ok:
            stats[f] += 1
        else:
            allf = False
    if allf:
        exact += 1
    else:
        wrong = [f for f in FIELDS
                 if not ((isinstance(gold.get(f), (int, float)) and not isinstance(gold.get(f), bool)
                          and isinstance(pred.get(f), (int, float)) and not isinstance(pred.get(f), bool)
                          and abs(float(gold[f]) - float(pred[f])) < 1e-4)
                         or pred.get(f) == gold.get(f))]
        fails.append((msgs[1]['content'], out.split('<')[0][:90], wrong))

print(f"== {os.path.basename(ckpt)} on {os.path.basename(ds_path)} ({n} samples, {time.time()-t0:.0f}s) ==")
print(f"format_pass: {fmt}/{n}  exact_match: {exact}/{n} ({100*exact/n:.1f}%)")
for f in FIELDS:
    print(f"  {f:12s}: {100*stats[f]/n:5.1f}%")
print('--- failures (first 25) ---')
for u, o, w in fails[:25]:
    print(f"  U:{u!r} pred:{o!r} wrong:{w}")
