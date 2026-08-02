import os
import re
import json
import random
import shutil
import datetime
from trainer.validate_json import extract_tool_call

# 類別標準化映射表
CATEGORY_MAPPING = {
    "餐飲飲食": "飲食", "飲食": "飲食", "餐飲": "飲食",
    "交通出行": "交通", "交通": "交通",
    "醫療保健": "醫療", "醫療": "醫療",
    "教育學習": "教育", "教育": "教育",
    "休閒娛樂": "娛樂", "娛樂": "娛樂",
    "日常雜貨": "日常", "日常": "日常", "日常用品": "日常",
    "還款": "還款",
    "其他": "其他",
}

def parse_date_from_query(query, sample_idx):
    """
    依據時間詞與 sample_idx (作為 random seed 確保可重現) 解析 user 提示詞中的日期
    """
    rng = random.Random(sample_idx)
    ref_date = datetime.date(2026, 7, 20)
    
    # 1. 匹配 X月Y號/日
    match_md = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[號日]", query)
    if match_md:
        month = int(match_md.group(1))
        day = int(match_md.group(2))
        month = max(1, min(12, month))
        day = max(1, min(31, day))
        try:
            return f"2026-{month:02d}-{day:02d}"
        except ValueError:
            pass
            
    # 2. 匹配相對時間詞
    if any(w in query for w in ["今天", "今日", "這週", "本週", "剛剛"]):
        return "2026-07-20"
    elif any(w in query for w in ["昨天", "昨晚"]):
        return "2026-07-19"
    elif "前天" in query:
        return "2026-07-18"
    elif "明天" in query:
        return "2026-07-21"
    elif any(w in query for w in ["上週", "上禮拜"]):
        return "2026-07-13"
        
    # 3. 僅提及月份，如 "12月"
    match_m = re.search(r"(\d{1,2})\s*月", query)
    if match_m:
        month = int(match_m.group(1))
        month = max(1, min(12, month))
        # 隨機產生這月份的一天
        max_days = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }[month]
        day = rng.randint(1, max_days)
        return f"2026-{month:02d}-{day:02d}"
        
    if "上個月" in query:
        day = rng.randint(1, 30)
        return f"2026-06-{day:02d}"
    elif any(w in query for w in ["這個月", "本月"]):
        day = rng.randint(1, 20)
        return f"2026-07-{day:02d}"
    elif "下個月" in query:
        day = rng.randint(1, 31)
        return f"2026-08-{day:02d}"
        
    # 4. 隨機日期範圍：參考日期 ±3 個月 (2026-04-20 ~ 2026-10-20)
    delta_days = rng.randint(-91, 92)
    target_date = ref_date + datetime.timedelta(days=delta_days)
    return target_date.strftime("%Y-%m-%d")

def convert_sample(sample, sample_idx):
    messages = sample.get("messages", [])
    
    # 尋找與提取各角色內容
    system_idx = -1
    user_idx = -1
    assistant_idx = -1
    
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            system_idx = i
        elif msg["role"] == "user":
            user_idx = i
        elif msg["role"] == "assistant":
            assistant_idx = i
            
    if system_idx == -1 or user_idx == -1 or assistant_idx == -1:
        # 不合規的對話結構，直接返回
        return sample

    system_content = messages[system_idx]["content"]
    user_content = messages[user_idx]["content"]
    assistant_content = messages[assistant_idx]["content"]
    
    # 1. System Prompt 加上「今天是 2026-07-20。」
    if not system_content.startswith("今天是"):
        system_content = "今天是 2026-07-20。" + system_content
        messages[system_idx]["content"] = system_content
        
    # 2. 解析 User Query 中的時間詞
    date_str = parse_date_from_query(user_content, sample_idx)
    
    # 3. 提取 Assistant Tool Call 並更新類別與加入日期
    try:
        args = extract_tool_call(assistant_content)
        
        # 類別標準化
        category = args.get("category", "")
        if category in CATEGORY_MAPPING:
            args["category"] = CATEGORY_MAPPING[category]
            
        # 加入日期
        args["date"] = date_str
        
        # 重新打包成標準 JSON 格式的 Tool Call
        new_tool_call = {
            "name": "add_record",
            "args": args
        }
        new_assistant_content = f"<tool_call>{json.dumps(new_tool_call, ensure_ascii=False)}</tool_call>"
        messages[assistant_idx]["content"] = new_assistant_content
        
        # 更新 system prompt 中的 tool definition (若有 enum 列表也一併標準化)
        # 對舊分類在 system prompt 中的 enum 進行簡單替換
        for old_cat, new_cat in CATEGORY_MAPPING.items():
            system_content = system_content.replace(f'"{old_cat}"', f'"{new_cat}"')
        messages[system_idx]["content"] = system_content
        
    except Exception as e:
        print(f"警告: 解析第 {sample_idx} 筆 assistant content 失敗: {e}")
        
    return sample

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"檔案不存在，跳過: {file_path}")
        return
        
    bak_path = file_path + ".bak"
    shutil.copyfile(file_path, bak_path)
    print(f"已將原檔案備份至 {bak_path}")
    
    converted_samples = []
    with open(bak_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                new_sample = convert_sample(sample, idx)
                converted_samples.append(new_sample)
            except Exception as e:
                print(f"警告: 處理第 {idx} 行時發生錯誤: {e}")
                
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"成功轉換 {file_path}，共計 {len(converted_samples)} 筆資料。")

def main():
    dataset_dir = "./dataset"
    files = ["train_strata.jsonl", "test_strata.jsonl"]
    
    print("=== 開始轉換資料集中的日期與標準化類別 ===")
    for filename in files:
        path = os.path.join(dataset_dir, filename)
        process_file(path)
    print("=== 資料集轉換完畢 ===")

if __name__ == "__main__":
    main()
