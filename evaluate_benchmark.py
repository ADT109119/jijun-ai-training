"""
記帳模型 Tool-calling 量化評測基準 (Benchmark) — Protocol v2 per-task 版

P0 變更（2026-08-21）：
- per-task 化：dataset 行可帶 "task" 欄位（缺省 add_record），統計按 (task, difficulty) 分組
- 欄位比對 schema 驅動：依 label args 的 keys 逐欄比對（不再硬編 5 欄），
  新 tool 加入後評測腳本自動支援，無需改代碼
- 新增 tool_selection 指標（預測 tool name == label tool name）
- torch lazy import：--help / mock 模式無需 torch 環境
- --quant：量化版本標籤（純報告標註，q4_0 vs fp16 各自記錄）

守門指標（見 tools.schema.json guardrail）：
- add_record 格式 ≥97% / EM ≥36% / 金額 ≥90%
- 新 tool 格式 ≥95% / 工具選擇 ≥90%

執行：
    python evaluate_benchmark.py --model_path ./dummy --dataset ./dataset/test_strata.jsonl
    python evaluate_benchmark.py --model_path saves/v2_transfer/best.pt --dataset tests/v2_transfer/test.jsonl --quant q4_0
"""

import os
import json
import argparse
import time

from trainer.validate_json import (
    extract_tool_call,
    extract_tool_calls,
    validate_record,
    calculate_reward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="記帳模型 Tool-calling 量化評測基準 (Benchmark)")
    parser.add_argument("--model_path", type=str, required=True, help="訓練完成的 PyTorch 模型權重路徑")
    parser.add_argument("--base_model", type=str, default="jingyaogong/minimind-3", help="分詞器基礎模型名稱或路徑")
    parser.add_argument("--dataset", type=str, required=True, help="分層測試數據集 (.jsonl) 路徑")
    parser.add_argument("--quant", type=str, default="", help="量化版本標籤（純報告標註，例如 q4_0 / fp16）")
    return parser.parse_args()


def _first_call(args_text):
    """從 label/預測文字取第一個 tool call 的 (name, args)；解析失敗回 (None, {})。

    用 v2 extract_tool_calls（舊資料單 call 完全相容）。
    """
    try:
        calls = extract_tool_calls(args_text)
    except ValueError:
        return None, {}
    if not calls:
        return None, {}
    c = calls[0]
    if c["error"] or not c["args"]:
        return c["name"], {}
    return c["name"], c["args"]


def load_stratified_dataset(dataset_path):
    """載入分層數據集（per-task：task 欄位缺省 add_record）。"""
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                difficulty = data.get("difficulty_level", "Level-1 (Simple)")
                task = data.get("task", "add_record")

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

                label_name, label_args = _first_call(label_text)

                dataset.append({
                    "id": idx,
                    "system_prompt": system_prompt,
                    "query": query,
                    "label_text": label_text,
                    "label_name": label_name,
                    "label_args": label_args,
                    "task": task,
                    "difficulty": difficulty,
                })
            except Exception as e:
                print(f"警告: 讀取第 {idx} 行測試資料失敗: {e}")
    return dataset


def _fields_match(pred_args, label_args):
    """schema 驅動欄位比對：label args 的每個 key 都要匹配（數值容差 1e-4）。"""
    for key, expected in label_args.items():
        actual = pred_args.get(key)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return False
            if abs(float(actual) - float(expected)) >= 1e-4:
                return False
        elif actual != expected:
            return False
    return True


def _field_correct(pred_args, label_args, field):
    expected = label_args.get(field)
    actual = pred_args.get(field)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and abs(float(actual) - float(expected)) < 1e-4)
    return actual == expected


def _new_group():
    return {
        "total": 0,
        "format_pass": 0,
        "tool_selection": 0,
        "exact_match": 0,
        "field_correct": {},
        "reward_sum": 0.0,
        "latency_sum": 0.0,
    }


def run_evaluation(model_path, base_model, dataset_path, quant_label=""):
    print("=== 啟動記帳模型量化評測（Protocol v2 per-task）===")
    print(f"模型權重: {model_path}")
    print(f"測試數據: {dataset_path}")
    if quant_label:
        print(f"量化版本: {quant_label}")

    # 1. 載入模型（torch lazy import：mock 模式不需要 torch）
    model = None
    device = None
    tokenizer = None
    if model_path != "dummy" and os.path.exists(model_path):
        try:
            import torch
            from model.model import BookkeepingLM, ModelConfig
            from model.tokenizer import BookkeepingTokenizer

            print("正在載入分詞器...")
            tokenizer = BookkeepingTokenizer(base_model)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = ModelConfig(vocab_size=tokenizer.vocab_size)
            model = BookkeepingLM(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"成功載入模型，運行裝置: {device}")
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

    # 3. 統計按 (task, difficulty) 分組
    stats = {}
    for idx, item in enumerate(test_data):
        key = (item["task"], item["difficulty"])
        g = stats.setdefault(key, _new_group())
        g["total"] += 1

        start_time = time.time()
        if model is not None:
            prompt_str = (
                f"<|im_start|>system\n{item['system_prompt']}\n"
                f"</think>\n"
                f"<|im_start|>user\n{item['query']}\n"
                f"</think>\n"
                f"<|im_start|>assistant\n"
            )
            import torch

            input_ids = torch.tensor(
                tokenizer.encode(prompt_str, add_special_tokens=False), dtype=torch.long
            ).unsqueeze(0).to(device)
            generated_ids = []
            curr_input = input_ids
            for _ in range(128):
                with torch.no_grad():
                    logits_gen, _ = model(curr_input)
                next_token = torch.argmax(logits_gen[0, -1, :]).unsqueeze(0).unsqueeze(0)
                generated_ids.append(next_token.item())
                if next_token.item() == tokenizer.eos_token_id:
                    break
                curr_input = torch.cat([curr_input, next_token], dim=-1)
            output_text = tokenizer.decode(generated_ids)
        else:
            import random
            random.seed(idx)
            if item["difficulty"] == "Level-3 (Reasoning)" and random.random() < 0.15:
                output_text = '<tool_call>{"amount": 150, "category": "餐飲", "account": "信用卡"'
            else:
                output_text = item["label_text"]
        g["latency_sum"] += (time.time() - start_time) * 1000

        # 4. 校驗（v2 per-task）
        label_name = item["label_name"]
        label_args = item["label_args"]
        pred_name, pred_args = _first_call(output_text)

        if pred_name is not None:
            g["format_pass"] += 1
            if label_name is not None and pred_name == label_name:
                g["tool_selection"] += 1
            # 欄位正確率（schema 驅動：label 的每個非 description 欄位）
            for field in label_args:
                if field == "description":
                    continue
                if _field_correct(pred_args, label_args, field):
                    g["field_correct"][field] = g["field_correct"].get(field, 0) + 1
            # 完全一致 = tool 相同 + label 全部欄位匹配
            if label_name is not None and pred_name == label_name and _fields_match(pred_args, label_args):
                g["exact_match"] += 1

        # RL reward 目前僅對 add_record 有意義（validate_record 為 v1 add_record 專用）；
        # per-tool reward 設計屬 P1+（規劃書 Tier 4），其他 task 標記 n/a
        if item["task"] == "add_record":
            g["reward_sum"] += calculate_reward(output_text, label_args)
            g["reward_count"] = g.get("reward_count", 0) + 1

    # 5. 輸出報告（按 task 分組，再按 difficulty）
    print("\n" + "=" * 64)
    title = "記帳模型量化評測基準報告 (Protocol v2)"
    if quant_label:
        title += f" [{quant_label}]"
    print(title)
    print("=" * 64)

    tasks = sorted(set(t for t, _ in stats.keys()))
    total_all = format_all = exact_all = 0
    for task in tasks:
        print(f"\n【Task: {task}】")
        task_total = 0
        for (t, diff), g in sorted(stats.items()):
            if t != task:
                continue
            total, task_total = g["total"], task_total + g["total"]
            print(f"  {diff}")
            print(f"    樣本數: {total}")
            print(f"    格式通過率: {g['format_pass'] / total * 100:.2f}% ({g['format_pass']}/{total})")
            print(f"    工具選擇: {g['tool_selection'] / total * 100:.2f}%")
            print(f"    完全匹配率: {g['exact_match'] / total * 100:.2f}%")
            for field in sorted(g["field_correct"].keys()):
                print(f"    {field} 正確率: {g['field_correct'][field] / total * 100:.2f}%")
            if g.get("reward_count"):
                print(f"    平均 RL Reward: {g['reward_sum'] / g['reward_count']:.2f} / 4.0")
            else:
                print("    平均 RL Reward: n/a (per-tool reward 屬 P1+)")
            print(f"    平均推論延遲: {g['latency_sum'] / total:.1f} ms")
            print("  " + "-" * 56)
            total_all += total
            format_all += g["format_pass"]
            exact_all += g["exact_match"]

    print(f"\n總計 ({total_all} 筆，{len(tasks)} 個 task):")
    print(f"  總格式通過率: {format_all / total_all * 100:.2f}%")
    print(f"  總精準匹配率: {exact_all / total_all * 100:.2f}%")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    test_jsonl_path = "./dataset/test_strata.jsonl"
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")

    if not os.path.exists(test_jsonl_path):
        mock_data = [
            {
                "difficulty_level": "Level-1 (Simple)",
                "task": "add_record",
                "messages": [
                    {"role": "system", "content": "今天是 2026-07-20。你是一個記帳助理。"},
                    {"role": "user", "content": "吃麥當勞花了 150 元，付現金"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 150, "category": "餐飲", "account": "現金", "description": "麥當勞", "type": "expense", "date": "2026-07-20"}</tool_call>'},
                ],
            },
            {
                "difficulty_level": "Level-2 (Noise)",
                "task": "add_record",
                "messages": [
                    {"role": "system", "content": "今天是 2026-07-20。你是一個記帳助理。"},
                    {"role": "user", "content": "今天下雨搭計程車回家，花了 250 元，刷悠遊卡"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 250, "category": "交通", "account": "悠遊卡", "description": "計程車", "type": "expense", "date": "2026-07-20"}</tool_call>'},
                ],
            },
            {
                "difficulty_level": "Level-3 (Reasoning)",
                "task": "add_record",
                "messages": [
                    {"role": "system", "content": "今天是 2026-07-20。你是一個記帳助理。"},
                    {"role": "user", "content": "昨天領了上個月的家教薪水 5000 元，存入銀行帳戶"},
                    {"role": "assistant", "content": '<tool_call>{"amount": 5000, "category": "薪水", "account": "銀行帳戶", "description": "家教", "type": "income", "date": "2026-07-19"}</tool_call>'},
                ],
            },
            {
                "difficulty_level": "Level-1 (Simple)",
                "task": "transfer",
                "messages": [
                    {"role": "system", "content": "今天是 2026-07-20。你是一個記帳助理。"},
                    {"role": "user", "content": "從現金轉 3000 到玉山帳戶"},
                    {"role": "assistant", "content": '<tool_call>{"name": "transfer", "args": {"from_account": "現金", "to_account": "玉山帳戶", "amount": 3000, "date": "2026-07-20"}}</tool_call>'},
                ],
            },
        ]
        with open(test_jsonl_path, "w", encoding="utf-8") as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    args = parse_args()
    run_evaluation(args.model_path, args.base_model, args.dataset, args.quant)
