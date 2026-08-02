import json, time, torch
from model.model import BookkeepingLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer
from trainer.validate_json import extract_tool_call, validate_record

BASE_MODEL = "jingyaogong/minimind-3"
MODEL_PATH = "./saves/best_bookkeeping_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256

CATEGORIES = ["飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款", "薪水", "獎金", "零用錢", "兼職", "投資", "利息", "欠款回收", "其他"]
ACCOUNTS = ["現金", "信用卡", "悠遊卡", "一卡通", "街口支付", "LINE Pay", "Apple Pay", "Google Pay", "郵局帳戶", "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"]
TOOL_DEF = {"name": "add_record", "description": "新增一筆記帳記錄",
            "parameters": {"type": "object", "properties": {
                "amount": {"type": "number"}, "category": {"type": "string", "enum": CATEGORIES},
                "account": {"type": "string", "enum": ACCOUNTS}, "description": {"type": "string"},
                "type": {"type": "string", "enum": ["expense", "income"]},
                "date": {"type": "string"}}, "required": ["amount", "category", "account", "type", "date"]}}
FULL_PROMPT = f'今天是 2026-07-20。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(TOOL_DEF, ensure_ascii=False)}'

print(f"Device: {DEVICE}")
tokenizer = BookkeepingTokenizer(BASE_MODEL)
config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=MAX_SEQ_LEN)
model = BookkeepingLM(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE); model.eval()

def generate(query):
    prompt = (f"<|im_start|>system\n{FULL_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n")
    input_ids = torch.tensor(tokenizer.encode(prompt, max_length=MAX_SEQ_LEN, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = []
    curr = input_ids
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits, _ = model(curr)
        next_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        token_id = next_id.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        curr = torch.cat([curr, next_id], dim=-1)
    return tokenizer.decode(generated)

TESTS = [
    ("薪水", "老闆轉帳薪水 35000 元，銀行帳戶"),
    ("獎金", "公司上個月發了年終獎金 50000 元，入帳到國泰帳戶"),
    ("零用錢", "爸媽這週給了我 2000 元生活費"),
    ("兼職", "這週末幫人接案寫程式賺了 3000 元"),
    ("飲食", "中午跟同事吃火鍋，一個人 400 元，刷信用卡"),
    ("交通", "搭高鐵從台北到高雄花了 1400 元"),
    ("醫療", "今天去診所看感冒，掛號加藥費 300 元"),
    ("娛樂", "週末和朋友去看電影，兩張票 600 元"),
    ("教育", "報名了線上英文課程，一年 9000 元"),
    ("利息", "銀行定存到期，領到利息 1200 元"),
    ("投資", "賣出一些股票獲利了 8000 元"),
    ("欠款回收", "朋友昨天還我之前借他的 5000 元"),
    ("日常", "去超市買了衛生紙和洗衣精，花了 350 元"),
    ("還款", "用網銀繳信用卡帳單 3000 元"),
    ("外送", "用 Foodpanda 叫外送晚餐花了 250 元"),
]

for expected, q in TESTS:
    t0 = time.time()
    raw = generate(q)
    try:
        args = extract_tool_call(raw)
        is_valid, errors = validate_record(args)
        cat = args.get("category", "?")
        desc = args.get("description", "?")
        typ = args.get("type", "?")
        amt = args.get("amount", "?")
        match = "✔" if cat == expected or (expected == "外送" and cat == "飲食") else "✘"
        print(f"{match} 期望={expected:4s} 實際={cat:4s} {typ:6s} ${amt} | {desc} | valid={is_valid}")
    except Exception as e:
        print(f"✘ 期望={expected:4s} 解析失敗: {str(e)[:60]} | raw={raw[:80]}")
    print(f"   [{time.time()-t0:.1f}s]")
