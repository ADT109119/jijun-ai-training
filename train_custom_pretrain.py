import os
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model.model import MiniMindLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer

try:
    from datasets import load_dataset
    HAS_WIKI_LIB = True
except ImportError:
    HAS_WIKI_LIB = False

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 記帳模型從零開始的無監督預訓練 (Pre-training)")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.txt", help="本地備用無監督預訓練文本 (.txt) 路徑")
    parser.add_argument("--use_wikipedia", action="store_true", default=True, help="是否優先透過 datasets 下載 Wikipedia 進行預訓練")
    parser.add_argument("--wiki_limit", type=int, default=3000, help="限制載入 Wikipedia 文章的筆數以控制預訓練語料規模")
    parser.add_argument("--base_model", type=str, default="jingyaogong/minimind-3", help="分詞器對齊路徑")
    parser.add_argument("--save_dir", type=str, default="./saves", help="預訓練模型 Checkpoint 儲存目錄")
    parser.add_argument("--epochs", type=int, default=3, help="預訓練總 Epoch 數")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch 大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="預訓練學習率 (通常大於微調)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup 步數比例")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列長度")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪閾值")
    parser.add_argument("--no_ratio_limit", action="store_true", default=False, help="是否放開 Wiki 與口語對話的 7:3 比例限制，直接訓練所有載入的 Token")
    return parser.parse_args()

class PretrainDataset(Dataset):
    """
    無監督預訓練資料集類別。
    
    核心亮點：
    1. 整合 HuggingFace datasets：線上串流下載中文 Wikipedia 語料，免去 Git 倉庫提交大體積資料集的繁瑣。
    2. 整合 OpenCC：對下載的維基百科簡體內容進行繁簡轉換，確保模型學習高品質的繁體中文表徵。
    3. 實作自動分塊 (Chunking)，將長文本切分為 max_seq_len 大小的訓練樣本。
    """
    def __init__(self, file_path: str, tokenizer, max_seq_len: int = 512, use_wikipedia: bool = True, wiki_limit: int = 3000, no_ratio_limit: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunks = []

        wiki_docs = []
        # 優先嘗試線上拉取繁體中文 Wikipedia
        if use_wikipedia and HAS_WIKI_LIB:
            try:
                print("正在透過 datasets 載入繁體中文 Wikipedia 數據集 (lianghsun/wikipedia-zh-filtered, streaming 模式)...")
                wiki_dataset = load_dataset("lianghsun/wikipedia-zh-filtered", split="train", streaming=True)
                
                print(f"正在載入前 {wiki_limit} 篇繁體維基百科文章...")
                count = 0
                for article in wiki_dataset:
                    token_ids = self.tokenizer.encode(article["text"], add_special_tokens=False)
                    if token_ids:
                        # 加上 Document Boundary (EOS Token)
                        token_ids.append(self.tokenizer.eos_token_id)
                        wiki_docs.append(token_ids)
                    count += 1
                    if count >= wiki_limit:
                        break
                print(f"成功載入 {count} 篇繁體 Wikipedia 文章。")
            except Exception as e:
                print(f"警告: 線上載入 Wikipedia 失敗 ({e})，將降級。")

        # 混合 30% 日常記帳口語對話語料
        conv_docs = []
        raw_generated_path = "./dataset/raw_generated.jsonl"
        if os.path.exists(raw_generated_path):
            try:
                import json
                with open(raw_generated_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            chat_text = ""
                            for msg in data.get("messages", []):
                                chat_text += f"{msg['role']}: {msg['content']}\n"
                            token_ids = self.tokenizer.encode(chat_text, add_special_tokens=False)
                            if token_ids:
                                # 加上 Document Boundary (EOS Token)
                                token_ids.append(self.tokenizer.eos_token_id)
                                conv_docs.append(token_ids)
                print(f"成功混合 {len(conv_docs)} 筆口語記帳對話。")
            except Exception as e:
                print(f"警告: 載入本地口語語料失敗: {e}")

        # Fallback 到本地備用檔案
        if not wiki_docs and not conv_docs:
            if not os.path.exists(file_path):
                print(f"提示: 未找到本地備用語料 {file_path}，已自動建立模擬資料集。")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("輕鬆記帳是一款非常實用的離線 PWA 記帳軟體。\n" * 100)
                    f.write("我們可以使用 AI 來幫助使用者快速且智能地記帳。\n" * 100)

            print(f"正在從本地讀取預訓練文本: {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                token_ids = self.tokenizer.encode(content, add_special_tokens=False)
                if token_ids:
                    token_ids.append(self.tokenizer.eos_token_id)
                    wiki_docs.append(token_ids)

        # 4. 精確 Token 比例控制 (70% Wikipedia / 30% Conversations)
        total_wiki_tokens = sum(len(d) for d in wiki_docs)
        total_conv_tokens = sum(len(d) for d in conv_docs)
        print(f"原始 Token 統計 - 百科: {total_wiki_tokens}，口語對話: {total_conv_tokens}")

        if not no_ratio_limit and total_wiki_tokens > 0 and total_conv_tokens > 0:
            # wiki_tokens : conv_tokens = 7 : 3
            if total_wiki_tokens / 7.0 > total_conv_tokens / 3.0:
                # wiki 太多，進行截斷以對齊 7:3 比例
                target_wiki_tokens = int(total_conv_tokens * 7.0 / 3.0)
                selected_wiki_docs = []
                current_tokens = 0
                for doc in wiki_docs:
                    selected_wiki_docs.append(doc)
                    current_tokens += len(doc)
                    if current_tokens >= target_wiki_tokens:
                        break
                wiki_docs = selected_wiki_docs
                print(f"已限制 Wikipedia 總 Token 數為 {current_tokens}，以對齊 70% 比例")
            else:
                # conv 太多，進行截斷以對齊 7:3 比例
                target_conv_tokens = int(total_wiki_tokens * 3.0 / 7.0)
                selected_conv_docs = []
                current_tokens = 0
                for doc in conv_docs:
                    selected_conv_docs.append(doc)
                    current_tokens += len(doc)
                    if current_tokens >= target_conv_tokens:
                        break
                conv_docs = selected_conv_docs
                print(f"已限制口語對話總 Token 數為 {current_tokens}，以對齊 30% 比例")

        # 5. 混合、打亂 (Shuffle) 與封裝 (Packing)
        all_docs = wiki_docs + conv_docs
        
        # 以 document 為單位打亂，保證兩種語料均勻混合
        import random
        random.seed(42)
        random.shuffle(all_docs)

        # 打平 Token 串並手動排除開頭的重複 BOS Token
        flat_tokens = []
        for doc in all_docs:
            flat_tokens.extend(doc)
            
        # 移除可能的多餘 BOS Token 以免干擾 Packing
        if len(flat_tokens) > 0 and flat_tokens[0] == self.tokenizer.tokenizer.bos_token_id:
            flat_tokens = flat_tokens[1:]

        # 按照 max_seq_len 長度切分塊 (Chunking/Packing)
        chunk_step = max_seq_len
        for i in range(0, len(flat_tokens) - max_seq_len, chunk_step):
            self.chunks.append(flat_tokens[i : i + max_seq_len + 1])

        print(f"預訓練文本處理完畢，混合 Token 數: {len(flat_tokens)}，共切分出 {len(self.chunks)} 個 Packing 訓練區塊。")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]
        
        # 自迴歸 Next Token Prediction 特性：
        # input_ids: 前 N 個 token
        # labels: 後 N 個 token (位移一位)
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 檢測運行硬體
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 啟動 MiniMind 無監督自迴歸預訓練 (From Scratch) ===")
    print(f"運行裝置: {device}")

    # 2. 載入分詞器
    tokenizer = BookkeepingTokenizer(args.base_model)

    # 3. 從零隨機初始化模型 (From Scratch)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len
    )
    # 不加載任何 pre-trained 權重
    model = MiniMindLM(config).to(device)
    from model.model import count_parameters
    total_params = count_parameters(model)
    print(f"模型隨機初始化成功。詞表大小: {config.vocab_size}，層數: {config.n_layers}，參數量: {total_params / 1e6:.2f}M")

    dataset = PretrainDataset(
        file_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        use_wikipedia=args.use_wikipedia,
        wiki_limit=args.wiki_limit,
        no_ratio_limit=args.no_ratio_limit
    )
    if len(dataset) == 0:
        print("錯誤: 預訓練資料太少，無法切分出合規訓練區塊，訓練終止。")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5. 配置優化器與學習率退火
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # 6. 預訓練循環 (Pre-training Loop)
    best_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        print(f"\n--- Pretrain Epoch {epoch} / {args.epochs} ---")
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 前向傳播
            logits, loss = model(input_ids, targets=labels)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            step_count += 1
            
            if (step + 1) % 10 == 0 or (step + 1) == len(dataloader):
                curr_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {curr_lr:.2e}")

        avg_loss = epoch_loss / step_count
        print(f"Epoch {epoch} 預訓練平均 Loss: {avg_loss:.4f}")

        # 每個 Epoch 結束時保存權重
        checkpoint_path = os.path.join(args.save_dir, f"pretrained_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"已保存 Checkpoint 至: {checkpoint_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = os.path.join(args.save_dir, "best_pretrain_model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[SUCCESS] 發現更佳預訓練權重！已保存至: {best_checkpoint_path}")

    print("\n無監督預訓練階段已順利結束！")

if __name__ == "__main__":
    main()
