import json
import torch
from torch.utils.data import Dataset

class BookkeepingDataset(Dataset):
    """
    自訂記帳資料集類別，繼承自 PyTorch Dataset。
    
    核心亮點：
    1. 實作 ChatML 對話模板序列化。
    2. 採用「Label Masking (Loss Masking)」策略，將 system prompt 與 user query 位置的 label 設為 -100。
       確保模型在全量 SFT 微調時，只對 assistant 輸出的 tool call 欄位計算損失與更新梯度，
       這能有效防止小模型遺忘預訓練語言基礎，並大幅加快收斂速度。
    """
    
    # 定義 ChatML 標記常數
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512, data_format: str = "json"):
        """
        Args:
            jsonl_path: jsonl 資料集路徑
            tokenizer: BookkeepingTokenizer 實例
            max_seq_len: 模型最大序列長度
            data_format: 資料格式 ("json" 或 "compressed")
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data_format = data_format.lower()
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"資料集 {jsonl_path} 載入成功，資料格式: {self.data_format}，樣本總數: {len(self.samples)} 筆。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = sample.get("messages", [])

        # 1. 拆分 system, user, assistant 的 content
        system_content = ""
        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        # 2. 如果為壓縮格式，則將 assistant_content 轉換為特殊標記協定
        if self.data_format == "compressed":
            try:
                from trainer.validate_json import extract_tool_call
                args = extract_tool_call(assistant_content)
                assistant_content = self.tokenizer.format_compressed_record(
                    amount=args.get("amount", 0),
                    category=args.get("category", ""),
                    account=args.get("account", ""),
                    description=args.get("description", ""),
                    record_type=args.get("type", "expense")
                )
            except Exception:
                # 容錯降級：若提取失敗，保留原 assistant_content
                pass

        # 3. 依據 ChatML 格式拼接對話字串，並分段進行 Token 處理
        prompt_str = (
            f"{self.IM_START}system\n{system_content}{self.IM_END}\n"
            f"{self.IM_START}user\n{user_content}{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        response_str = f"{assistant_content}{self.IM_END}"

        # 編碼 prompt 與 response (add_special_tokens 預設為 False，手動拼接最安全)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)

        # 4. 進行左截斷優化以保全 Assistant Response
        total_len = len(prompt_ids) + len(response_ids)
        if total_len > self.max_seq_len:
            response_len = len(response_ids)
            if response_len >= self.max_seq_len:
                # 極端情況：僅 response 就超長，只保留 response 部分的截斷
                response_ids = response_ids[:self.max_seq_len]
                prompt_ids = []
            else:
                # 截斷 prompt 的最左側 (保留右側最接近對話的部分)
                allowed_prompt_len = self.max_seq_len - response_len
                prompt_ids = prompt_ids[-allowed_prompt_len:]

        # 5. 重新拼接與建立 labels 陣列並進行 Loss Masking
        input_ids = prompt_ids + response_ids
        
        # Loss Masking: logits[t] 預測 input[t+1]，so target[t] = input[t+1]
        # prompt 部分用 -100 填滿，僅對 response 部分計算損失
        labels = [-100] * max(0, len(prompt_ids) - 1) + response_ids + [-100]

        # 7. 動態 Padding 至最大長度 (對齊 batch 長度，或固定為 max_seq_len)
        padding_len = self.max_seq_len - len(input_ids)
        if padding_len > 0:
            # input_ids 用 pad_token_id 填充，labels 用 -100 填充
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len

        # 8. 轉換為 PyTorch Tensor 格式
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

if __name__ == "__main__":
    # 測試 Dataset 的編碼與 Label Masking 效果
    from model.tokenizer import BookkeepingTokenizer
    try:
        # 本地載入分詞器
        tokenizer = BookkeepingTokenizer("jingyaogong/minimind-3")
        
        # 隨機產生一個暫存 jsonl 用於測試
        test_path = "./dataset/test_dataset.jsonl"
        with open(test_path, "w", encoding="utf-8") as f:
            mock_sample = {
                "messages": [
                    {"role": "system", "content": "你是一個記帳助理。"},
                    {"role": "user", "content": "剛吃午餐花了 150 元付現"},
                    {"role": "assistant", "content": "<tool_call>{\"amount\": 150}</tool_call>"}
                ]
            }
            f.write(json.dumps(mock_sample, ensure_ascii=False) + "\n")
            
        dataset = BookkeepingDataset(test_path, tokenizer, max_seq_len=64)
        sample = dataset[0]
        
        print("\n=== Dataset 測試 ===")
        print("input_ids 總長度:", len(sample["input_ids"]))
        print("labels 總長度:", len(sample["labels"]))
        
        # 檢查 -100 被遮蔽的比例
        unmasked_count = (sample["labels"] != -100).sum().item()
        print("計算 Loss (非 -100) 的 Token 數量:", unmasked_count)
        
        # 還原計算 Loss 的 Token 內容
        loss_tokens = [id.item() for id in sample["labels"] if id.item() != -100]
        print("模型學習生成的預測目標文字:", tokenizer.decode(loss_tokens))
        
    except Exception as e:
        print("測試載入失敗 (通常是本地尚未建立 tokenizer 環境):", e)
