import json
import time
import gradio as gr
import torch

from model.model import BookkeepingLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer
from trainer.validate_json import extract_tool_call, validate_record

BASE_MODEL = "jingyaogong/minimind-3"
MODEL_PATH = "./saves/best_bookkeeping_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256

CATEGORIES = [
    "飲食", "日常", "交通", "娛樂", "醫療", "教育", "還款",
    "薪水", "獎金", "零用錢", "兼職", "投資", "利息", "欠款回收",
    "其他"
]
ACCOUNTS = [
    "現金", "信用卡", "悠遊卡", "一卡通", "街口支付",
    "LINE Pay", "Apple Pay", "Google Pay", "郵局帳戶",
    "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"
]
TOOL_DEF = {
    "name": "add_record", "description": "新增一筆記帳記錄",
    "parameters": {
        "type": "object", "properties": {
            "amount": {"type": "number"},
            "category": {"type": "string", "enum": CATEGORIES},
            "account": {"type": "string", "enum": ACCOUNTS},
            "description": {"type": "string"},
            "type": {"type": "string", "enum": ["expense", "income"]},
            "date": {"type": "string", "description": "ISO 8601 格式日期，例如 YYYY-MM-DD"}
        },
        "required": ["amount", "category", "account", "type", "date"]
    }
}
FULL_PROMPT = f'今天是 2026-07-20。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(TOOL_DEF, ensure_ascii=False)}'

PROMPTS = {
    "完整工具定義（推薦）": FULL_PROMPT,
    "簡潔助理": FULL_PROMPT,
}

print(f"Device: {DEVICE}")
print("Loading tokenizer...")
tokenizer = BookkeepingTokenizer(BASE_MODEL)
print(f"Vocab size: {tokenizer.vocab_size}")

print("Loading model...")
config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=MAX_SEQ_LEN)
model = BookkeepingLM(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("Model loaded.")


def generate(query: str, system_prompt: str) -> str:
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = torch.tensor(
        tokenizer.encode(prompt, max_length=MAX_SEQ_LEN, add_special_tokens=False),
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

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


def resolve_prompt(prompt_key: str, custom_prompt: str) -> str:
    return custom_prompt.strip() if custom_prompt.strip() else PROMPTS.get(prompt_key, FULL_PROMPT)

def inference(query: str, prompt_key: str, custom_prompt: str, progress=gr.Progress()):
    if not query.strip():
        return "", "", "", "", "", ""

    system_prompt = resolve_prompt(prompt_key, custom_prompt)

    progress(0.2, desc="Generating...")
    t0 = time.time()
    raw = generate(query, system_prompt)
    latency = (time.time() - t0) * 1000

    progress(0.7, desc="Parsing...")
    try:
        args = extract_tool_call(raw)
        parsed = json.dumps(args, ensure_ascii=False, indent=2)
        is_valid, errors = validate_record(args)

        errors_text = "; ".join(errors) if errors else ""
        type_zh = "收入" if args.get("type") == "income" else "支出"
        pretty = (
            f"**類別:** {args.get('category', 'N/A')}\n"
            f"**金額:** {args.get('amount', 'N/A')} 元\n"
            f"**帳戶:** {args.get('account', 'N/A')}\n"
            f"**類型:** {type_zh}\n"
            f"**描述:** {args.get('description', 'N/A')}\n\n"
            f"**格式驗證:** {'✅ 通過' if is_valid else '❌ ' + errors_text}"
        )
    except ValueError as e:
        args = None
        parsed = f"解析失敗: {e}"
        pretty = "**格式驗證:** ❌ 無法解析 tool_call"

    format_pass_str = "✅ 通過" if args else "❌ 失敗"
    latency_str = f"{latency:.0f} ms"

    return raw, parsed, pretty, format_pass_str, latency_str, system_prompt


with gr.Blocks(title="記帳模型測試 Demo") as demo:
    gr.Markdown(
        "# 記帳 Tool-calling 模型測試\n"
        "輸入自然語言記帳描述，模型會自動解析為結構化工具呼叫。",
    )

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(
                label="輸入記帳描述",
                placeholder="例如：吃麥當勞花了 150 元，付現金",
                lines=3,
            )
            prompt_key = gr.Dropdown(
                label="System Prompt 模板",
                choices=["完整工具定義（推薦）", "簡潔助理"],
                value="完整工具定義（推薦）",
            )
            custom_prompt = gr.Textbox(
                label="自訂 System Prompt（選填，留空則使用上方模板）",
                placeholder="輸入自訂提示詞...",
                lines=2,
            )
            current_system_prompt = gr.Textbox(
                label="目前使用的 System Prompt",
                value=FULL_PROMPT,
                lines=4,
                interactive=False,
            )
            submit = gr.Button("送出", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["吃麥當勞花了 150 元，付現金"],
                    ["昨天領了上個月的家教薪水 5000 元，存入銀行帳戶"],
                    ["今天下雨搭計程車回家，花了 250 元，刷悠遊卡"],
                    ["買了杯珍珠奶茶 55 元"],
                    ["網購書本花了 380 元，用信用卡付款"],
                    ["加油花了 1200 元，刷信用卡"],
                    ["老闆轉帳薪水 35000 元，銀行帳戶"],
                    ["上週買顯卡花了 15000 元，用信用卡"],
                    ["前天寵物看病 800 元，付現金"],
                ],
                inputs=[query],
                cache_examples=False,
            )

        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    format_pass = gr.Textbox(label="格式校驗", interactive=False)
                with gr.Column(scale=1):
                    latency = gr.Textbox(label="推論延遲", interactive=False)

            with gr.Tabs():
                with gr.TabItem("解析結果"):
                    result = gr.Markdown("等待輸入...")
                with gr.TabItem("原始輸出"):
                    raw_output = gr.Code(label="Raw Text", interactive=False)
                with gr.TabItem("JSON"):
                    json_output = gr.Code(label="Parsed JSON", language="json", interactive=False)

    def update_prompt_display(prompt_key, custom_prompt):
        return resolve_prompt(prompt_key, custom_prompt)

    prompt_key.change(
        fn=update_prompt_display,
        inputs=[prompt_key, custom_prompt],
        outputs=[current_system_prompt],
    )
    custom_prompt.change(
        fn=update_prompt_display,
        inputs=[prompt_key, custom_prompt],
        outputs=[current_system_prompt],
    )

    submit.click(
        fn=inference,
        inputs=[query, prompt_key, custom_prompt],
        outputs=[raw_output, json_output, result, format_pass, latency, current_system_prompt],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
