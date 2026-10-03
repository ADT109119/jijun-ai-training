import json
import os
import re

ACCOUNT_KEYWORDS = [
    '現金', '付現', '信用卡', '刷卡', '卡', '悠遊卡', '一卡通', 'icash', '悠遊付',
    '街口', 'LINE Pay', 'Apple Pay', 'Google Pay', 'Samsung Pay', '台灣Pay', '全支付',
    '全盈', 'OPEN錢包', 'Pi錢包', '橘子支付', '歐付寶', '郵局', '銀行', '存款', '帳戶',
    '外幣', '加密貨幣', 'Richart', 'LINE Bank', '將來銀行', '樂天', 'O-Bank',
    '轉帳', '匯款', '扣款', '提款', '交割戶', '複委託', '美元', '日圓', '台新', '國泰',
    '富邦', '中信', '玉山', '永豐', '聯邦', '元大', '星展', '滙豐', '兆豐', '第一', '華南',
    '彰化', '合庫', '渣打', '大戶', 'iLeo', 'KOKO', 'SnY'
]

def user_mentions_account(user_text: str) -> bool:
    """檢查使用者輸入的文本中是否明確含有任何支付媒介關鍵字"""
    for kw in ACCOUNT_KEYWORDS:
        if kw in user_text:
            return True
    return False

def update_assistant_tool_call(assistant_content: str, new_account: str = "現金") -> str:
    """更新 assistant 的 <tool_call> JSON 中的 account 欄位為 new_account"""
    m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", assistant_content)
    if not m:
        try:
            data = json.loads(assistant_content.strip())
            if isinstance(data, dict):
                if "args" in data and isinstance(data["args"], dict):
                    data["args"]["account"] = new_account
                elif "account" in data:
                    data["account"] = new_account
                return f"<tool_call>{json.dumps(data, ensure_ascii=False)}</tool_call>"
        except Exception:
            return assistant_content
        return assistant_content

    json_str = m.group(1).strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            if "args" in data and isinstance(data["args"], dict):
                data["args"]["account"] = new_account
            elif "account" in data:
                data["account"] = new_account
            new_json_str = json.dumps(data, ensure_ascii=False)
            return f"<tool_call>{new_json_str}</tool_call>"
    except Exception:
        pass
    return assistant_content

def process_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"檔案不存在: {filepath}")
        return

    updated_samples = []
    fixed_count = 0
    total_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_count += 1
            sample = json.loads(line)
            msgs = sample.get("messages", [])

            user_msg_idx = -1
            asst_msg_idx = -1
            user_text = ""

            for i, msg in enumerate(msgs):
                if msg.get("role") == "user":
                    user_msg_idx = i
                    user_text = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    asst_msg_idx = i

            if user_msg_idx != -1 and asst_msg_idx != -1:
                if not user_mentions_account(user_text):
                    asst_content = msgs[asst_msg_idx].get("content", "")
                    new_content = update_assistant_tool_call(asst_content, "現金")
                    if new_content != asst_content:
                        msgs[asst_msg_idx]["content"] = new_content
                        fixed_count += 1

            updated_samples.append(sample)

    with open(filepath, "w", encoding="utf-8") as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"=== 已處理 {filepath} ===")
    print(f"  - 總筆數: {total_count}")
    print(f"  - 修復（未提及帳戶設為現金）筆數: {fixed_count} ({fixed_count/total_count*100:.2f}%)\n")

def main():
    target_files = [
        "dataset/raw_generated.jsonl",
        "dataset/train_strata.jsonl",
        "dataset/test_strata.jsonl"
    ]
    for tf in target_files:
        process_file(tf)

if __name__ == "__main__":
    main()
