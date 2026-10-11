# T7 短語強化 — v4/v4b/v4c 資料與訓練管線修復（2026-10-11）

## 背景

使用者回報具體失敗案例：輸入「早餐60」模型產出錯誤金額。路線圖（
ai-expansion-plan.md）外的插入戰役。追查過程揭露兩個訓練管線級 bug
（date 欄位不可學、金額湊整先驗），影響遠超短語本身。

前置狀態：v1 模型已發佈 HF `the-walking-fish/jijun-LM`（safetensors）；
本機 `.pt` 權重因磁碟遷移遺失，需先建立「HF → 本機 .pt」返還鏈路才能
continue-SFT。

## 本輪發現與修復（按根因排序）

### 1. `dataset.py` 日期硬編碼 → date 欄位學習被徹底摧毀（已修復，v4b）

`BookkeepingDataset` 對任何以「今天是 YYYY-MM-DD」開頭的 system prompt
一律改寫成固定「今天是 2026-07-20。」。後果：訓練時模型所見日期恆定，
gold label 卻是各種真實日期——模型被要求擬合一個與輸入無關的條件分布，
`date` 欄位在有效訊號意義上不可學。v1 報的 date 61% 是評測端僥倖
（大量樣本 gold=當天且 prompt 錨定當天），continue-SFT 後崩到近 0%。

- 修復：僅在 system prompt 缺日期前綴時才補上真實 ref_date，絕不改寫既有日期。
- 驗證：v4c `test_v3` date **89.5%**（v4 修復前 0.2%）。

### 2. 金額湊整先驗 → 短語型輸入抄錯數字（已修復，v4c）

「早餐60」→150 的真正根因**不是** few-shot 範例污染（第一假設，dump
prompt 後推翻：system prompt 無任何 few-shot 範例）。實際是兩個資料設計
缺陷疊加：
1. 訓練集所有金額被湊整成十/百倍數 → 模型學會「金額=整十整百」，遇到
   60 這種值傾向輸出該品項訓練分布中的典型整數；
2. v4 產生器刻意讓 test 商品詞與 train 零重疊 →「早餐」對模型是零經驗
   詞，只能瞎猜品項典型金額。

v4c 對策（`gen/build_v4c.py`）：
- 商品詞全數入訓練池（早餐/網費/保險費/家教/借款回收等錯例詞全涵蓋）；
- 取樣規則 40% 典型區間 / 35% 全區間 / **25% 破先驗極端值**（早餐 10 元、
  牛排 20 元這種「該品項不可能」的金額）；
- 65% 湊整 + **35% 保留非整數**，打掉「金額必為整十百」的捷徑；
- 金額書寫補小數萬（7.69萬）/複合千（3千5）/2萬5千 等形勢。

### 3. 泛化探針取代「背答案式」測試集

v4 的 test 商品詞零重疊設計看似嚴格，其實同時剝奪了學習機會（詞都沒見過，
category 只能瞎猜），無法區分「抄寫機制學會」與「背訓練分布」。v4c 改用
**雙測試集**：
- `regression_v4.jsonl`（103）：原 short_test，生產回歸錨點（含「早餐60」精確案例）；
- `probe_unseen.jsonl`（160）：20 個全新商品詞（按摩/貓飼料/稿費/房東退押金…），
  專測抄寫機制。結果 **amount 90.6% / date 93.8%** = 機制真的學會了；
  category 30% 屬預期（無語義先驗可猜，58M 上限內）。

### 4. HF safetensors → 本機 .pt 返還鏈路（新增 `restore_hf_checkpoint.py`）

- 權重名映射：Llama 命名（`model.layers.N.self_attn.q_proj` …）→ 本機自訂命名
  （`layers.N.attention.wq` …），共 111 tensors。
- Tokenizer 相容性驗證（`tmp_tokcheck.py`，一次性）：HF vocab 6406 = 本機
  BookkeepingTokenizer 6406，特殊 Token ID 零錯位，150 條編碼零差異 →
  continue-SFT 安全。
- 教訓：`--max_seq_len` 必須 **實測** token 長度分布後決定（v4 量得 p50=554 /
  p90=589 / max=627，故 512 會截斷 system prompt 的 enum 池與規則文字；
  改 768）。runbook 早就寫了「先量測再決定」，這次是親身體證。

### 5. 訓練腳本 VRAM 治理（`train_custom_sft.py`）

使用者回報訓練時 VRAM 全滿並吃到 GPU 共用記憶體（Windows WDDM 溢出 → 桌面卡頓）。
根因：FP32 訓練 + bs16×768 activation 疊上桌面本身佔用，12GB 卡不夠。
- 新增 `--amp bf16`（預設開）：autocast 混合精度，激活值顯存減半，4070S 原生支援，
  loss 曲線與 FP32 無實質差異（冒煙測試對照）。
- 新增 `--vram_fraction`（預設 0.6）：`torch.cuda.set_per_process_memory_fraction`
  進程級硬上限，超過在訓練進程內報 OOM，**不會**溢出到共用記憶體。
- 實測（bf16+bs8）：Step 正常收斂，整卡含桌面峰值 ~8.9GB / 12.3GB。

## 訓練紀錄（起點皆為 `saves/jijun_lm_restored.pt`）

| run | 資料 | 關鍵設定 | val loss | 結論 |
|---|---|---|---|---|
| v4 | train_v4 5502 | bs16 lr5e-5 **msl 512** | 0.1504 | prompt 截斷（enum 池被切），date 仍 0%（dataset bug 未修） |
| v4b | train_v4 5502 | bs16 lr5e-5 msl 768 + dataset 修復 | 0.0902 | date 復活（75–90%），短語 EM 35%；剩餘失敗=金額湊整先驗 |
| v4c | train_v4c 7902 | bs8 bf16 lr5e-5 msl 768 | **0.0721** | 短語 EM **75.7%**，破先驗策略奏效；epoch3 收斂 epoch4 持平無過擬合 |

## 評測（greedy，`eval_checkpoints.py`；詳表見 README 6.1）

- regression_v4 (103)：EM 0%→35%→**75.7%**，category/description/type 100%
- probe_unseen (160)：amount **90.6%** / date **93.8%**（抄寫機制驗證）
- test_v3 (493)：date 0%→**89.5%**，amount 95.3%，無災難性遺忘
- gate_sub300 (300)：amount 96.0% / account 98.7% / date 75.3%

## 已知殘留（定性，非阻斷）

1. 短語集 account ~12% 失敗多為測試集隨機指定非現金帳戶但句子無支付線索，
   模型默猜現金在產品語意上更合理 → 評測集噪音。
2. `43k`/`1.13萬` 小數複合金額零星抄錯（`還款入帳入帳1.13萬` 雙「入帳」等
   變異形勢覆蓋仍不足）。
3. probe category 30%：全新詞無先驗，要再提升需更大模型或更多同義詞族譜。
4. 長句 description 逐字 EM ~27% 為自由文字比對必然，PWA 不驗證該欄位。

## 下一步

- 導出 v4c → ONNX/GGUF（`export_model.py` 鏈路），PWA 端實測「早餐60」端到端。
- 若繼續 v5：小數複合金額書寫擴量 + account 線索一致性檢查（資料產生器
  保證「無支付線索 ⇒ gold=現金」，消掉評測噪音）。
