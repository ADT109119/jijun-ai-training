import os
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model.model import MiniMindLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer
from dataset import BookkeepingDataset
from trainer.validate_json import extract_tool_call, validate_record

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 離線記帳模型全量微調 (Full SFT)")
    parser.add_argument("--train_path", type=str, default="./dataset/train_strata.jsonl", help="訓練集 JSONL 路徑")
    parser.add_argument("--val_path", type=str, default="./dataset/test_strata.jsonl", help="驗證集 JSONL 路徑")
    parser.add_argument("--base_model", type=str, default="jingyaogong/minimind-3", help="基礎模型名稱或路徑")
    parser.add_argument("--save_dir", type=str, default="./saves", help="模型 Checkpoint 儲存目錄")
    parser.add_argument("--epochs", type=int, default=8, help="微調的總 Epoch 數")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch 大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="學習率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 步數比例")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列長度")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪閾值")
    parser.add_argument("--format", type=str, default="json", choices=["json", "compressed"], help="資料格式 (json 或 compressed)")
    parser.add_argument("--eval_generate", action="store_true", default=False, help="是否在驗證集進行文本生成評測")
    parser.add_argument("--max_eval_samples", type=int, default=5, help="生成評測的最大樣本數")
    parser.add_argument("--pretrained_path", type=str, default=None, help="預訓練權重路徑 (.pt)，若不指定則從隨機初始化開始訓練")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing 係數 (預設 0.0 = 不使用)")
    parser.add_argument("--n_layers", type=int, default=None, help="覆寫 Transformer 層數")
    parser.add_argument("--d_model", type=int, default=None, help="覆寫隱藏層維度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率 (預設 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (預設 0.1)")
    return parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    餘弦退火學習率調度器，帶有 Warmup 預熱階段
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def run_evaluation(model, dataloader, tokenizer, device, eval_generate=False, max_eval_samples=5, label_smoothing=0.0):
    """
    驗證評估：計算 Loss 的同時，整合前端 JSON 解析器統計 JSON_Format_Pass_Rate
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    format_pass_count = 0
    valid_count = 0
    total_samples = 0
    eval_gen_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向傳播計算 loss
            logits, loss = model(input_ids, targets=labels, label_smoothing=label_smoothing)
            total_loss += loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()

            # 解碼模型生成 (限制樣本數，防止評測卡死)
            if eval_generate and eval_gen_count < max_eval_samples:
                for i in range(input_ids.size(0)):
                    if eval_gen_count >= max_eval_samples:
                        break
                    total_samples += 1
                    
                    # 找到 labels 中不是 -100 的區間 (即模型需要預測生成的部分)
                    label_mask = labels[i] != -100
                    if not label_mask.any():
                        continue
                    
                    # 取得 prompt token id
                    prompt_len = label_mask.nonzero()[0].item()
                    prompt_ids = input_ids[i][:prompt_len].unsqueeze(0)
                    
                    # 簡易自迴歸生成
                    generated_ids = []
                    curr_input = prompt_ids
                    for _ in range(128):  # 限制生成最大 token
                        logits_gen, _ = model(curr_input)
                        next_token_logits = logits_gen[0, -1, :]
                        
                        # 貪婪解碼 (Greedy Decoding)
                        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                        generated_ids.append(next_token.item())
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        curr_input = torch.cat([curr_input, next_token], dim=-1)
                    
                    # 解碼為文字
                    generated_text = tokenizer.decode(generated_ids)
                    eval_gen_count += 1
                    
                    # 整合 PWA 驗證器校驗
                    try:
                        args = extract_tool_call(generated_text)
                        format_pass_count += 1
                        
                        is_valid, _ = validate_record(args)
                        if is_valid:
                            valid_count += 1
                    except ValueError:
                        pass

    avg_loss = total_loss / max(1, total_tokens)
    format_pass_rate = (format_pass_count / total_samples) * 100 if total_samples > 0 else 0.0
    valid_record_rate = (valid_count / total_samples) * 100 if total_samples > 0 else 0.0
    
    return avg_loss, format_pass_rate, valid_record_rate

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 初始化設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 啟動 MiniMind 全量微調 (Full SFT) ===")
    print(f"運行裝置: {device}")
    print(f"訓練格式: {args.format}")

    # 2. 載入分詞器
    tokenizer = BookkeepingTokenizer(args.base_model)
    
    # 3. 載入模型配置與主模型
    config_kwargs = dict(vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len)
    if args.n_layers is not None:
        config_kwargs['n_layers'] = args.n_layers
    if args.d_model is not None:
        config_kwargs['d_model'] = args.d_model
    config_kwargs['dropout'] = args.dropout
    config = ModelConfig(**config_kwargs)
    model = MiniMindLM(config).to(device)
    
    # 若有指定預訓練權重，則載入到隨機初始化的模型上
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            state = torch.load(args.pretrained_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"已載入預訓練權重: {args.pretrained_path}")
        else:
            print(f"警告: 預訓練權重 {args.pretrained_path} 不存在，使用隨機初始化")
    
    from model.model import count_parameters
    total_params = count_parameters(model)
    print(f"模型初始化成功。詞表大小對齊: {config.vocab_size}，層數: {config.n_layers}，參數量: {total_params / 1e6:.2f}M")

    # 4. 準備訓練集與驗證集 DataLoader
    train_dataset = BookkeepingDataset(args.train_path, tokenizer, args.max_seq_len, data_format=args.format)
    val_dataset = BookkeepingDataset(args.val_path, tokenizer, args.max_seq_len, data_format=args.format)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. 配置優化器與 Cosine 學習率調度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    best_format_rate = -1.0
    best_val_loss = float("inf")

    # 6. 訓練主迴圈
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        print(f"\n--- Epoch {epoch} / {args.epochs} ---")
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 前向傳播
            logits, loss = model(input_ids, targets=labels, label_smoothing=args.label_smoothing)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，預防微小模型全量梯度更新時崩潰
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            step_count += 1
            
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                curr_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {curr_lr:.2e}")

        avg_train_loss = epoch_loss / step_count
        print(f"Epoch {epoch} 訓練集平均 Loss: {avg_train_loss:.4f}")

        # 7. 驗證與格式校驗
        print("正在進行驗證集評估與 PWA JSON 格式檢驗...")
        val_loss, format_pass_rate, valid_record_rate = run_evaluation(
            model, val_loader, tokenizer, device,
            eval_generate=args.eval_generate,
            max_eval_samples=args.max_eval_samples,
            label_smoothing=args.label_smoothing
        )
        
        print(f"驗證集指標:")
        print(f"  - 平均 Loss: {val_loss:.4f}")
        print(f"  - JSON 格式通過率 (JSON_Format_Pass_Rate): {format_pass_rate:.2f}%")
        print(f"  - 欄位合規率 (Schema_Valid_Rate)      : {valid_record_rate:.2f}%")

        # 8. 動態保存 Best Checkpoint (以格式通過率為首要指標)
        # 這能確保我們選出的模型，在前端 PWA WASM 運行時最不容易發生格式毀損
        is_best_format = format_pass_rate > best_format_rate
        is_best_loss = (abs(format_pass_rate - best_format_rate) < 1e-4) and (val_loss < best_val_loss)
        
        if is_best_format or is_best_loss:
            best_format_rate = format_pass_rate
            best_val_loss = val_loss
            
            checkpoint_path = os.path.join(args.save_dir, "best_bookkeeping_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[SUCCESS] 發現更優模型！已保存至: {checkpoint_path} (Format Pass: {best_format_rate:.2f}%)")
            
            # 同步保存一份 HF 格式設定與權重方便後續 ONNX / GGUF 匯出
            # model.save_pretrained(...) 等邏輯可在 Spike 後對齊

    # 9. 無論如何，最後一個 Epoch 的模型也儲存一份 (方便 label_smoothing 等未收斂情況)
    last_ckpt = os.path.join(args.save_dir, "last_bookkeeping_model.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"[INFO] 最後 Epoch 模型已保存至: {last_ckpt}")

if __name__ == "__main__":
    main()
