# P1 SFT v2.0 Windows 訓練 Runbook

> 日期：2026-08-21｜前置：WSL 端資料管線已完成（dataset/v2/*.jsonl 已備妥）
> 機器：Windows（venv 在 training repo 下，GPU 訓練機）

## 0. 前置確認

```powershell
cd <training-repo>
# 1. 資料已在位（從 WSL 複製或共用磁碟）：
dir dataset\v2\train_v2.jsonl   # 應 ~7400 筆（3323 舊 + ~4100 新）
dir dataset\v2\test_v2.jsonl    # 應 ~1850 筆（831 舊 + ~1000 新）
# 2. v1 模型權重（continue-SFT 起點）：
dir saves\best_bookkeeping_model.pt
# 3. 快速 sanity：token 長度檢查（新 v2 prompt 較長，確認 max_seq_len 夠用）
python -c "
import json, sys
from model.tokenizer import BookkeepingTokenizer
tok = BookkeepingTokenizer('jingyaogong/minimind-3')
lens = []
for line in open('dataset/v2/train_v2.jsonl', encoding='utf-8'):
    s = json.loads(line)
    m = {x['role']: x['content'] for x in s['messages']}
    full = m.get('system','') + m.get('user','') + m.get('assistant','')
    lens.append(len(tok.encode(full, add_special_tokens=False)))
lens.sort()
import statistics
print(f'n={len(lens)} p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}')
print('建議 max_seq_len:', '1024 足夠' if lens[-1] <= 1020 else '需評估截斷影響')
"
```

**關鍵**：若 p95 接近 1024，左截斷（dataset.py）會吃掉 system prompt 的 tools 定義 → 新 tool 學習失敗。若發生，選項：
- (a) 把 v2 訓練的 max_seq_len 提到 1280（需 model.py RoPE 上限配合，model.py L17 預設 1024，可用 `--max_seq_len 1280` 若 ModelConfig 支援）
- (b) 縮短 prompt（減少 account enum 數量）
**先量測再決定，不要憑空猜。**

## 1. 訓練（continue-SFT from v1 best）

```powershell
python train_custom_sft.py ^
  --train_path dataset\v2\train_v2.jsonl ^
  --val_path  dataset\v2\test_v2.jsonl ^
  --pretrained_path saves\best_bookkeeping_model.pt ^
  --format json ^
  --epochs 4 ^
  --lr 5e-5 ^
  --batch_size 8 ^
  --max_seq_len 1024 ^
  --save_dir saves\v2 ^
  --eval_generate ^
  --max_eval_samples 20
```

**超參與 v1 差異（有意為之）**：
- `--pretrained_path` 用 **best_bookkeeping_model.pt**（v1 SFT 成品，非 pretrain 底模）— continue-SFT
- `--epochs 4`（v1 用 8）：continue-SFT 過擬合風險高，少跑
- `--lr 5e-5`（v1 用 2e-4）：保護既有 add_record 能力，防災難性遺忘

若 epoch 1 後 val 指標開始惡化 → 直接用 epoch 1 的 checkpoint。

## 2. 評測（per-task + 守門指標）

```powershell
# 新 v2 測試集（per-task 指標）
python evaluate_benchmark.py --model_path saves\v2\best_bookkeeping_model.pt --dataset dataset\v2\test_v2.jsonl --quant fp16

# 守門：v1 原始測試集（add_record 基線不能掉）
python evaluate_benchmark.py --model_path saves\v2\best_bookkeeping_model.pt --dataset dataset\test_strata.jsonl --quant fp16
```

**守門指標（tools.schema.json guardrail）**：
| 指標 | 守門線 | v1 基線 |
|------|--------|---------|
| add_record 格式 | ≥97% | 98.68% |
| add_record EM | ≥36% | 39.35% |
| add_record 金額 | ≥90% | 92.30% |
| transfer 格式 | ≥95% | — (新) |
| transfer 工具選擇 | ≥90% | — (新) |
| add_debt/settle_debt 格式 | ≥95% | — (新) |

**任何守門線未達 → 停。回調 lr（改 2e-5）或 epochs（改 3）重試；仍不行 → 增加 add_record 混合比例到 60%。**

## 3. GGUF 量化 + 上傳

```powershell
python export_all_quants.py --ckpt_path saves\v2\best_bookkeeping_model.pt --out_dir exports\v2
# 上傳 q4_0 到 HF（the-walking-fish/jijun-LM-GGUF），ETag 機制讓用戶端自動更新
```

## 4. 主專案端接線（Windows 完成後，回 WSL 做）

aiService.js 目前 `generateSystemPrompt` 仍是 v1 單工具。v2 上線時：
1. `parseRecord` 改用 `parseToolCalls`（已在 toolsSchema.js）
2. `generateSystemPrompt` 改傳 4-tool 陣列
3. UI 預覽卡（多筆確認）— P1 範圍外，先只接 transfer/debt 單筆
4. E2E 測試：用 demo_gguf.py 打 v2 模型 20 筆 transfer/debt 看實際格式率

## 風險清單
- **格式混雜**：v1 舊資料是單工具 prompt、新資料是 4-tool prompt。模型要學會「依 prompt 裡的 tool 數量決定輸出」。若新 tool 格式率低但 add_record 正常 → 可能是 prompt 長度吃掉 tool 定義（看 §0 token 量測）。
- **disambiguation 干擾**：transfer「轉帳」vs add_record「還款」vs settle_debt「還錢」語意重疊。Level-3 資料量若不足，disambiguation 正確率會低 — 屬預期，先觀察。
- **continue-SFT 遺忘**：lr 過大或 epochs 過多 → add_record 掉。守門指標就是抓這個。
