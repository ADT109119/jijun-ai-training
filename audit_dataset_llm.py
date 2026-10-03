import os
import sys
import json
import re
import random
import asyncio
import argparse
import time
from typing import Dict, Any, Tuple, Optional
from openai import AsyncOpenAI, APIError, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 強制 Windows 控制台標準輸出為 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

AUDITOR_SYSTEM_PROMPT = """你是一個嚴格的 AI 資料集質檢與對齊稽核員。你的任務是審核每一筆「中文口語記帳與 Tool Call 對話軌跡」是否符合嚴格的品質與對齊標準。

【檢查規則與對齊標準】
1. 金額 (amount) 數值與邏輯：
   - 金額必須完全符合 user 描述的真實交易金額。若涉及推理計算（如「買 3 本 100 元的書」），金額必須計算精準 (300)。
   - 不得出現金額計算錯誤、數字截斷或無中生有的巨額數字幻覺。

2. 分類 (category) 與收支類型 (type)：
   - category 必須精準符合 user 消費/收入實情（例如吃午餐歸「飲食」、搭捷運歸「交通」、領薪水歸「薪水」）。
   - type 必須與消費/收入屬性完全一致（expense 支出 或 income 收入），切勿混用。

3. 帳戶 (account) 預設與提取規則（核心對齊規則）：
   - 若 user 明確提到支付管道（例：「刷信用卡」、「用 LINE Pay 付款」、「郵局轉帳」），account 必須精準對應。
   - 若 user 【完全未提及】任何支付方式/管道（例：「午餐 99元」、「捷運35」、「全聯牛奶85元」），assistant 的 account 欄位【必須為『現金』】。若未提及卻出現非現金帳戶（例如『Google Pay』、『台新信用卡』、『icash』等），一律視為帳戶幻覺，判定不通過。

4. 日期 (date) 計算與預設規則：
   - 若 user 提及相對時間（如「昨天」、「上週三」、「前天」），date 必須以 system prompt 中的參考日期（今天是 YYYY-MM-DD）為基準計算出精準的 ISO 日期。
   - 若 user 未提及時間，date 必須與 system prompt 中的參考日期完全一致。

5. 語法與品質合規性：
   - user 內容必須符合繁體中文口語語法，禁止包含亂碼、簡體怪異詞彙或無意義字元。
   - assistant 的 tool_call 必須結構完整、格式合規且欄位齊全。

【輸出格式要求】
你可以在輸出中進行思考推理過程。
但在回應的最後，請務必輸出格式嚴格的最終判定行：

判定：通過

或

判定：不通過 | 原因：<寫明不通過的簡短具體原因>
"""

def extract_audit_result(llm_response: str) -> Tuple[bool, str]:
    """從 LLM 回應中解析最終判定 (通過/不通過) 與原因"""
    if not llm_response:
        return False, "模型回應內容為空"

    # 剔除 <think>...</think> 區塊
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", llm_response).strip()

    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    if not lines:
        lines = [llm_response.strip()]

    # 從最後幾行倒著搜尋判定標籤
    for line in reversed(lines[-5:]):
        if "判定：不通過" in line or "判定: 不通過" in line or "不通過" in line:
            reason_match = re.search(r"原因[：:]\s*(.*)", line)
            reason = reason_match.group(1).strip() if reason_match else line
            return False, reason
        if "判定：通過" in line or "判定: 通過" in line or (line == "通過") or ("通過" in line and "不通過" not in line):
            return True, "質檢合格"

    # 備用檢查全文
    if "不通過" in cleaned:
        return False, "全文包含不通過判定"
    if "通過" in cleaned:
        return True, "質檢合格"

    return False, f"無法解析最終判定行，原始回應片段: {cleaned[-100:]}"

async def audit_single_sample(client: AsyncOpenAI, model_name: str, sample: Dict[str, Any], thinking_budget: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
    """向 LLM 發起對單筆樣本的稽核請求"""
    messages = sample.get("messages", [])
    system_text = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_text = next((m["content"] for m in messages if m.get("role") == "user"), "")
    asst_text = next((m["content"] for m in messages if m.get("role") == "assistant"), "")

    formatted_input = (
        f"【待稽核樣本內容】\n"
        f"1. System Context: {system_text}\n"
        f"2. User 輸入: {user_text}\n"
        f"3. Assistant 輸出 (Tool Call): {asst_text}\n\n"
        f"請依據稽核標準進行審查，並在最後一行輸出『判定：通過』或『判定：不通過 | 原因：...』"
    )

    history = [
        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": formatted_input}
    ]

    max_retries = 6
    for attempt in range(max_retries):
        try:
            extra_body = None
            if "qwen" in model_name.lower():
                if thinking_budget and thinking_budget > 0:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": thinking_budget}}
                else:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response = await client.chat.completions.create(
                model=model_name,
                messages=history,
                temperature=0.1,
                max_tokens=1024,
                stream=False,
                extra_body=extra_body
            )

            raw_text = response.choices[0].message.content or ""
            is_pass, reason = extract_audit_result(raw_text)
            return is_pass, reason, sample

        except RateLimitError:
            wait_sec = (2 ** attempt) + random.uniform(1.0, 3.0)
            await asyncio.sleep(wait_sec)
        except APIError:
            await asyncio.sleep(2.0)
        except Exception:
            await asyncio.sleep(2.0)

    return False, "API 連線或重試超時", sample

async def run_audit(input_file: str, kept_file: str, rejected_file: str, model_name: str, api_url: str, api_key: str, concurrency: int = 4, thinking_budget: int = 0):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到輸入檔案: {input_file}")

    # 建立輸出目錄
    os.makedirs(os.path.dirname(os.path.abspath(kept_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(rejected_file)), exist_ok=True)

    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass

    total_samples = len(samples)
    print(f"=== LLM 資料集逐行稽核開始 ===")
    print(f"  - 輸入檔案: {input_file} (共 {total_samples} 筆)")
    print(f"  - 通過輸出: {kept_file}")
    print(f"  - 不通過輸出: {rejected_file}")
    print(f"  - 稽核 LLM 模型: {model_name}")
    print(f"  - 並行數: {concurrency}\n")

    import httpx
    limits = httpx.Limits(max_connections=concurrency * 10, max_keepalive_connections=concurrency * 2)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_url,
        http_client=httpx.AsyncClient(limits=limits)
    )

    # 確保清空舊檔
    with open(kept_file, "w", encoding="utf-8") as f:
        pass
    with open(rejected_file, "w", encoding="utf-8") as f:
        pass

    passed_count = 0
    rejected_count = 0
    processed_count = 0
    t0 = time.time()

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(sample, idx):
        nonlocal passed_count, rejected_count, processed_count
        async with semaphore:
            is_pass, reason, sample_obj = await audit_single_sample(client, model_name, sample, thinking_budget)
            
            processed_count += 1
            if is_pass:
                passed_count += 1
                with open(kept_file, "a", encoding="utf-8") as f_kept:
                    f_kept.write(json.dumps(sample_obj, ensure_ascii=False) + "\n")
            else:
                rejected_count += 1
                rej_obj = {"reason": reason, "sample": sample_obj}
                with open(rejected_file, "a", encoding="utf-8") as f_rej:
                    f_rej.write(json.dumps(rej_obj, ensure_ascii=False) + "\n")

            if processed_count % 10 == 0 or processed_count == total_samples:
                elapsed = time.time() - t0
                speed = processed_count / elapsed if elapsed > 0 else 0
                pass_rate = (passed_count / processed_count) * 100
                print(f"進度: {processed_count}/{total_samples} ({processed_count/total_samples*100:.1f}%) | 通過: {passed_count} | 退件: {rejected_count} | 通過率: {pass_rate:.1f}% | 速度: {speed:.1f} 筆/秒")

    tasks = [worker(sample, idx) for idx, sample in enumerate(samples)]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    final_pass_rate = (passed_count / total_samples * 100) if total_samples > 0 else 0
    print(f"\n=== 稽核完成 ===")
    print(f"  - 總費時: {elapsed:.1f} 秒")
    print(f"  - 稽核總筆數: {total_samples}")
    print(f"  - ✅ 通過筆數: {passed_count} ({final_pass_rate:.2f}%) -> 已存至 {kept_file}")
    print(f"  - ❌ 退件筆數: {rejected_count} ({100 - final_pass_rate:.2f}%) -> 已存至 {rejected_file}")

def main():
    parser = argparse.ArgumentParser(description="使用 LLM 逐行對記帳資料集進行質檢與對齊稽核")
    parser.add_argument("--input", type=str, default="dataset/train_strata.jsonl", help="輸入欲稽核的 dataset JSONL 路徑")
    parser.add_argument("--kept_out", type=str, default="dataset/train_audited.jsonl", help="稽核通過的輸出 JSONL 路徑")
    parser.add_argument("--rejected_out", type=str, default="dataset/rejected_audited.jsonl", help="稽核退件的輸出 JSONL 路徑")
    parser.add_argument("--api_url", type=str, default=os.environ.get("OPENAI_BASE_URL"), help="API 端點 (可設於 .env)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="API 金鑰 (可設於 .env)")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "vllm/Qwen3.6-27B"), help="稽核 LLM 模型名稱 (可設於 .env)")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", 4)), help="並行請求數")
    parser.add_argument("--thinking_budget", type=int, default=0, help="推理模型的 thinking_budget (0 = 關閉/預設)")

    args = parser.parse_args()

    if not args.api_key or not args.api_url:
        raise ValueError("[安全錯誤] 未設定 OPENAI_API_KEY 或 OPENAI_BASE_URL！請檢查 .env 檔案。")

    asyncio.run(run_audit(
        args.input,
        args.kept_out,
        args.rejected_out,
        args.model,
        args.api_url,
        args.api_key,
        args.concurrency,
        args.thinking_budget
    ))

if __name__ == "__main__":
    main()
