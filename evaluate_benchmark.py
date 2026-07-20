import os
import json
import argparse
import time
import torch
from model.model import MiniMindLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer
from trainer.validate_json import extract_tool_call, validate_record, calculate_reward

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 記帳 Tool-calling 量化評測基準 (Benchmark)")
    parser.add_argument("--model_path", type=str, required=True, help="微調後的 PyTorch 模型權重路徑")
    parser.add_argument("--base_model", type=str, default="jingyaogong/minimind-3", help="基礎 model/tokenizer 路徑")
    parser.add_argument("--dataset", type=str, required=True, help="分層測試數據集 (.jsonl) 路徑")
    return parser.parse_args()

def load_stratified_dataset(dataset_path):
    """
    載入分層數據集，並根據 difficulty_level 進行分組
    """
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # 取得難度等級 (預設為 Level-1 簡單，若資料中無此欄位)
                difficulty = data.get("difficulty_level", "Level-1 (Simple)")
                
                # 從對話格式 messages 中提取 user query 與 assistant label
                messages = data.get("messages", [])
                query = ""
                label_text = ""
                system_prompt = "你是一個記帳助理。"
                for msg in messages:
                    if msg["role"] == "system":
                        system_prompt = msg["content"]
                    elif msg["role"] == "user":
                        query = msg["content"]
                    elif msg["role"] == "assistant":
                        label_text = msg["content"]
                
                # 從 label 中還原成 args dict 供比對
                try:
                    label_args = extract_tool_call(label_text)
                except ValueError:
                    label_args = {}

                dataset.append({
                    "id": idx,
                    "system_prompt": system_prompt,
                    "query": query,
                    "label_text": label_text,
                    "label_args": label_args,
                    "difficulty": difficulty
                })
            except Exception as e:
                print(f"警告: 讀取第 {idx} 行測試資料失敗: {e}")
    return dataset

def run_evaluation(model_path, base_model, dataset_path):
    print("=== 啟動 MiniMind 離線記帳小模型量化評測 ===")
    print(f"模型權重: {model_path}")
    print(f"測試數據: {dataset_path}")
    
    # 1. 載入 Tokenizer 與模型
    print("正在載入分詞器...")
    tokenizer = BookkeepingTokenizer(base_model)
    
    # 這裡預留 PyTorch 模型的加載，Spike 驗證後可加載微調權重
    # 由於在無 GPU 環境或初次執行時可能無法加載權重，此處在加載失敗時提供 Mock 推論模式
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 支援 dummy 路徑來跑 mock 模式，或是嘗試加載真實模型
    if model_path != "dummy" and os.path.exists(model_path):
        try:
            config = ModelConfig(vocab_size=tokenizer.vocab_size)
            model = MiniMindLM(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"成功加載模型，運行裝置: {device}")
        except Exception as e:
            print(f"警告: 未能加載實體模型 ({e})，評測將使用模擬推論模式進行模擬評測。")
            model = None
    else:
        print("提示: 模型路徑不存在或為 'dummy'，啟動模擬評測模式。")
    
    # 2. 載入測試數據
    test_data = load_stratified_dataset(dataset_path)
    if not test_data:
        print("錯誤: 沒有可用的測試語料，評測終止。")
        return
        
    print(f"載入測試語料成功，共計 {len(test_data)} 筆。")
    
    # 統計指標初始化
    stats = {}
    
    # 3. 逐筆進行評估
    for idx, item in enumerate(test_data):
        difficulty = item["difficulty"]
        if difficulty not in stats:
            stats[difficulty] = {
                "total": 0,
                "format_pass": 0,
                "exact_match": 0,
                "amount_correct": 0,
                "category_correct": 0,
                "account_correct": 0,
                "reward_sum": 0.0,
                "latency_sum": 0.0
            }
            
        diff_stats = stats[difficulty]
        diff_stats["total"] += 1
        
        # 模擬/執行推論
        start_time = time.time()
        
        # 實際推論邏輯 (Mock 或 Torch 推論)
        if model is not None:
            prompt_str = (
                f"<|im_start|>system\n{item['system_prompt']}<|im_end|>\n"
                f"<|im_start|>user\n{item['query']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            input_ids = torch.tensor(tokenizer.encode(prompt_str, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
            
            generated_ids = []
            curr_input = input_ids
            for _ in range(128):
                with torch.no_grad():
                    logits_gen, _ = model(curr_input)
                next_token_logits = logits_gen[0, -1, :]
                
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                generated_ids.append(next_token.item())
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                curr_input = torch.cat([curr_input, next_token], dim=-1)
                
            output_text = tokenizer.decode(generated_ids)
        else:
            # 使用與前端對齊的 mock 進行評測
            import random
            random.seed(idx) # 確保評測結果可重現
            
            if difficulty == "Level-3 (Reasoning)" and random.random() < 0.15:
                # 模擬 JSON 毀損 (括號未閉合)
                output_text = '<tool_call>{"amount": 150, "category": "餐飲", "account": "信用卡"'
            elif difficulty == "Level-3 (Reasoning)" and random.random() < 0.20:
                # 模擬金額提取出錯
                output_text = item["label_text"].replace(str(item["label_args"].get("amount", 0)), "NaN")
            else:
                output_text = item["label_text"]
                
        latency = (time.time() - start_time) * 1000 # 毫秒
        diff_stats["latency_sum"] += latency
        
        # 4. 校驗模型輸出
        label_args = item["label_args"]
        
        # 格式通過率驗證
        try:
            pred_args = extract_tool_call(output_text)
            diff_stats["format_pass"] += 1
            
            # 驗證金額
            is_valid, _ = validate_record(pred_args)
            if is_valid:
                # 比對金額
                if abs(float(pred_args.get("amount", 0)) - float(label_args.get("amount", 0))) < 1e-4:
                    diff_stats["amount_correct"] += 1
                # 比對分類
                if pred_args.get("category") == label_args.get("category"):
                    diff_stats["category_correct"] += 1
                # 比對帳戶
                if pred_args.get("account") == label_args.get("account"):
                    diff_stats["account_correct"] += 1
                    
                # 是否完全一致
                if (abs(float(pred_args.get("amount", 0)) - float(label_args.get("amount", 0))) < 1e-4 and
                    pred_args.get("category") == label_args.get("category") and
                    pred_args.get("account") == label_args.get("account") and
                    pred_args.get("type") == label_args.get("type")):
                    diff_stats["exact_match"] += 1
        except ValueError:
            pass # 格式毀損，各欄位皆算失敗
            
        # 計算 RL Reward 增強分數
        reward = calculate_reward(output_text, label_args)
        diff_stats["reward_sum"] += reward

    # 5. 輸出精美的學術評測報告
    print("\n" + "="*60)
    print("                MINIMIND 量化評測基準報告")
    print("="*60)
    
    grand_total = 0
    grand_format = 0
    grand_exact = 0
    
    for diff, data in stats.items():
        total = data["total"]
        format_pass_rate = (data["format_pass"] / total) * 100
        exact_match_rate = (data["exact_match"] / total) * 100
        avg_latency = data["latency_sum"] / total
        avg_reward = data["reward_sum"] / total
        
        grand_total += total
        grand_format += data["format_pass"]
        grand_exact += data["exact_match"]
        
        print(f"難度分層: {diff}")
        print(f"  - 測試樣本數: {total} 筆")
        print(f"  - 格式通過率: {format_pass_rate:.2f}% ({data['format_pass']}/{total})")
        print(f"  - 完全匹配率: {exact_match_rate:.2f}% ({data['exact_match']}/{total})")
        print(f"  - 金額正確率: {(data['amount_correct'] / total * 100):.2f}%")
        print(f"  - 分類正確率: {(data['category_correct'] / total * 100):.2f}%")
        print(f"  - 帳戶正確率: {(data['account_correct'] / total * 100):.2f}%")
        print(f"  - 平均 RL Reward 分數: {avg_reward:.2f} / 4.0")
        print(f"  - 平均推論延遲: {avg_latency:.2f} ms")
        print("-" * 60)
        
    print(f"總計統計 ({grand_total} 筆測試樣本):")
    print(f"  - 總格式通過率 (Format Pass Rate): {(grand_format / grand_total * 100):.2f}%")
    print(f"  - 總精準匹配率 (Exact Match Rate) : {(grand_exact / grand_total * 100):.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 用於本地測試的假 dataset 生成，確保評測腳本開箱即用
    test_jsonl_path = "./dataset/test_strata.jsonl"
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")
        
    if not os.path.exists(test_jsonl_path):
        # 寫入少量模擬分層測試數據
        mock_data = [
            {
                "difficulty_level": "Level-1 (Simple)",
                "messages": [
                    {"role": "user", "content": "吃麥當勞花了 150 元，付現金"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 150, "category": "餐飲", "account": "現金", "description": "麥當勞", "type": "expense"}</tool_call>'}
                ]
            },
            {
                "difficulty_level": "Level-2 (Noise)",
                "messages": [
                    {"role": "user", "content": "今天下雨搭計程車回家，花了 250 元，刷悠遊卡"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 250, "category": "交通", "account": "悠遊卡", "description": "計程車", "type": "expense"}</tool_call>'}
                ]
            },
            {
                "difficulty_level": "Level-3 (Reasoning)",
                "messages": [
                    {"role": "user", "content": "昨天領了上個月的家教薪水 5000 元，存入銀行帳戶"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 5000, "category": "薪水", "account": "銀行帳戶", "description": "家教", "type": "income"}</tool_call>'}
                ]
            }
        ]
        with open(test_jsonl_path, "w", encoding="utf-8") as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    # 執行評測
    # 範例執行：python evaluate_benchmark.py --model_path ./dummy --dataset ./dataset/test_strata.jsonl
    args = parse_args()
    run_evaluation(args.model_path, args.base_model, args.dataset)
