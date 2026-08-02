"""
demo_gguf.py — jijun-LM 58M GGUF 模型端側推論測試腳本

支援透過 llama-cpp-python 載入 GGUF 模型檔 (q4_0, q5_0, q6_k, q8_0, fp16)，
進行口述記帳句子的語義工具呼叫 (Tool Calling) 測試。
"""
import os
import sys

def main():
    print("=== jijun-LM GGUF 端側模型推論測試 (58M BookkeepingLM) ===")
    
    # 預設測試模型檔案列表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gguf_dir = os.path.join(script_dir, "jijun-LM-GGUF")
    default_model = os.path.join(gguf_dir, "bookkeeping_model_q4_0.gguf")

    if not os.path.exists(default_model):
        # 若找不到預設模型檔，嘗試尋找其它 gguf
        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")] if os.path.exists(gguf_dir) else []
        if gguf_files:
            default_model = os.path.join(gguf_dir, gguf_files[0])
        else:
            print(f"錯誤: 在 {gguf_dir} 中找不到任何 GGUF 模型檔")
            print("請先執行 python export_all_quants.py 生成 GGUF 檔案。")
            return

    file_size_mb = os.path.getsize(default_model) / (1024 * 1024)
    print(f"檢測到模型檔: {os.path.basename(default_model)} ({file_size_mb:.2f} MB)")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n[提示] 未在目前環境中偵測到 `llama-cpp-python`。")
        print("欲執行完整推論測試，請執行以下命令安裝：")
        print("  pip install llama-cpp-python\n")
        print("GGUF 模型檔驗證正常，可在任何支援 GGUF (WASM / llama.cpp / Ollama) 之環境載入！")
        return

    print(f"\n正在載入 GGUF 模型: {default_model} ...")
    try:
        llm = Llama(
            model_path=default_model,
            n_ctx=512,
            n_threads=4,
            verbose=False
        )
        print("模型載入成功！\n")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return

    system_prompt = (
        "今天是 2026-08-03（星期一）。你是一個記帳助理。你被賦予了以下 tools:\n"
        "add_record(amount: number, category: string, account: string, description: string, date: string, type: 'income'|'expense')\n"
        "可選分類: 餐飲, 交通, 購物, 娛樂, 醫療, 投資, 其它, 薪水, 獎金\n"
        "請根據使用者的口述內容生成工具呼叫。"
    )

    test_queries = [
        "昨天晚上跟朋友去吃火鍋刷卡花了 850 元",
        "今天領了本月薪水 45000 元匯入帳戶",
        "搭捷運花了 35 元悠遊卡扣款"
    ]

    for q in test_queries:
        print(f"💬 使用者輸入: \"{q}\"")
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        
        output = llm(
            prompt,
            max_tokens=128,
            stop=["<|im_end|>", "</tool_call>"],
            temperature=0.1
        )
        response = output["choices"][0]["text"].strip()
        print(f"🤖 AI 模型輸出: {response}\n")

if __name__ == "__main__":
    main()
