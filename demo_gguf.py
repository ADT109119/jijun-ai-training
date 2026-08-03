import os
import sys
import json
import time
import argparse
import gradio as gr
from trainer.validate_json import extract_tool_call, validate_record

# 強制 Windows 控制台標準輸出為 UTF-8 避免文字編碼顯示問題
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_DIR = os.path.join(SCRIPT_DIR, "jijun-LM-GGUF")

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
FULL_PROMPT = f'今天是 2026-08-03（星期一）。你是一個記帳助理。你被賦予了以下 tools:\n{json.dumps(TOOL_DEF, ensure_ascii=False)}'

# 模型快取字典
MODEL_CACHE = {}

def get_available_models():
    if not os.path.exists(GGUF_DIR):
        return {}
    models = {}
    files = sorted(os.listdir(GGUF_DIR), key=lambda x: 0 if "q4_0" in x else 1)
    for f in files:
        if f.endswith(".gguf"):
            full_path = os.path.join(GGUF_DIR, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            models[f"{f} ({size_mb:.1f} MB)"] = full_path
    return models

def load_gguf_model(model_path):
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("未安裝 `llama-cpp-python` 套件，請執行 `pip install llama-cpp-python`")

    print(f"正在載入 GGUF 模型: {model_path} ...")
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_batch=512,
        verbose=False
    )
    MODEL_CACHE[model_path] = llm
    return llm

def generate_gguf(llm, query: str, system_prompt: str) -> str:
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    output = llm.create_completion(
        prompt_text,
        max_tokens=128,
        temperature=0.1,
        stop=["<|im_end|>", "</tool_call>"]
    )
    return output["choices"][0]["text"].strip()

def inference(query: str, selected_model_name: str, custom_prompt: str, progress=gr.Progress()):
    if not query.strip():
        return "", "", "", "", "", ""

    system_prompt = custom_prompt.strip() if custom_prompt.strip() else FULL_PROMPT
    model_map = get_available_models()
    model_path = model_map.get(selected_model_name, list(model_map.values())[0] if model_map else "")

    if not model_path or not os.path.exists(model_path):
        return "錯誤", "找不到 GGUF 模型檔案", "", "❌ 失敗", "0 ms", system_prompt

    progress(0.2, desc="載入/呼叫 GGUF 模型...")
    t0 = time.time()
    try:
        llm = load_gguf_model(model_path)
        raw = generate_gguf(llm, query, system_prompt)
    except Exception as err:
        return f"模型推論錯誤: {err}", "", f"**錯誤:** {err}", "❌ 失敗", "0 ms", system_prompt
    
    latency = (time.time() - t0) * 1000

    progress(0.7, desc="解析 Tool Call ...")
    try:
        args = extract_tool_call(raw)
        parsed = json.dumps(args, ensure_ascii=False, indent=2)
        is_valid, errors = validate_record(args)

        errors_text = "; ".join(errors) if errors else ""
        type_zh = "收入" if args.get("type") == "income" else "支出"
        pretty = (
            f"**分類:** {args.get('category', 'N/A')}\n\n"
            f"**金額:** {args.get('amount', 'N/A')} 元\n\n"
            f"**帳戶:** {args.get('account', 'N/A')}\n\n"
            f"**收支:** {type_zh}\n\n"
            f"**描述:** {args.get('description', 'N/A')}\n\n"
            f"**日期:** {args.get('date', 'N/A')}\n\n"
            f"**格式驗證:** {'✅ 通過' if is_valid else '❌ ' + errors_text}"
        )
    except ValueError as e:
        args = None
        parsed = f"解析失敗: {e}"
        pretty = "**格式驗證:** ❌ 無法解析 tool_call"

    format_pass_str = "✅ 通過" if args else "❌ 失敗"
    latency_str = f"{latency:.0f} ms"

    return raw, parsed, pretty, format_pass_str, latency_str, system_prompt

def run_cli_interactive():
    print("=== jijun-LM GGUF 互動式命令行測試 (輸入 q 退出) ===")
    models = get_available_models()
    if not models:
        print("錯誤: 找不到 GGUF 模型檔案，請先執行 convert_hf_to_gguf_full.py")
        return

    model_path = list(models.values())[0]
    file_name = list(models.keys())[0]
    print(f"預設載入模型: {file_name}")

    try:
        llm = load_gguf_model(model_path)
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return

    while True:
        try:
            query = input("\n請輸入記帳描述 (或 q 退出): ").strip()
            if not query or query.lower() == 'q':
                print("程式已退出。")
                break
            
            t0 = time.time()
            raw = generate_gguf(llm, query, FULL_PROMPT)
            latency = (time.time() - t0) * 1000
            
            print(f"🤖 RAW 輸出: {raw}")
            try:
                args = extract_tool_call(raw)
                print(f"✅ 工具呼叫解析: {json.dumps(args, ensure_ascii=False)}")
            except Exception as e:
                print(f"❌ 工具呼叫解析失敗: {e}")
            print(f"⏱️ 推論延遲: {latency:.0f} ms")
        except (KeyboardInterrupt, EOFError):
            print("\n程式已退出。")
            break

def launch_web_ui():
    models_map = get_available_models()
    model_choices = list(models_map.keys())
    default_choice = model_choices[0] if model_choices else "未找到模型檔案"

    with gr.Blocks(title="jijun-LM GGUF 記帳模型測試 Demo") as demo:
        gr.Markdown(
            "# 輕鬆記帳 jijun-LM (GGUF 端側模型測試 Demo)\n"
            "輸入口述中文記帳描述，端側 GGUF LLM 將自動解析為結構化工具呼叫 (Tool Calling)。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(
                    label="輸入記帳描述",
                    placeholder="例如：昨天晚餐吃火鍋刷卡花了 850 元",
                    lines=3,
                )
                selected_model_name = gr.Dropdown(
                    label="選擇 GGUF 量化版本模型",
                    choices=model_choices,
                    value=default_choice,
                )
                custom_prompt = gr.Textbox(
                    label="自訂 System Prompt（選填，留空則使用預設系統提示）",
                    placeholder="輸入自訂提示詞...",
                    lines=2,
                )
                current_system_prompt = gr.Textbox(
                    label="目前使用的 System Prompt",
                    value=FULL_PROMPT,
                    lines=4,
                    interactive=False,
                )
                submit = gr.Button("🚀 送出推論", variant="primary", size="lg")

                gr.Examples(
                    examples=[
                        ["吃麥當勞花了 150 元，付現金"],
                        ["昨天晚餐吃火鍋刷卡花了 850 元"],
                        ["今天領了本月薪水 45000 元匯入帳戶"],
                        ["搭捷運花了 35 元悠遊卡扣款"],
                        ["網購書本花了 380 元，用信用卡付款"],
                        ["加油花了 1200 元，刷信用卡"],
                        ["上週買顯卡花了 15000 元，用信用卡"],
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
                    with gr.TabItem("原始輸出 (Raw)"):
                        raw_output = gr.Code(label="Raw Output", interactive=False)
                    with gr.TabItem("JSON"):
                        json_output = gr.Code(label="Parsed JSON", language="json", interactive=False)

        submit.click(
            fn=inference,
            inputs=[query, selected_model_name, custom_prompt],
            outputs=[raw_output, json_output, result, format_pass, latency, current_system_prompt],
        )

    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)

def main():
    parser = argparse.ArgumentParser(description="jijun-LM GGUF Demo 測試工具")
    parser.add_argument("--cli", action="store_true", help="開啟命令行互動模式而非 Web UI")
    args = parser.parse_args()

    if args.cli:
        run_cli_interactive()
    else:
        launch_web_ui()

if __name__ == "__main__":
    main()
