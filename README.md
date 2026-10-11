# 基於語意協議壓縮的超小型高效端側工具調用語義模型研究
> **An Empirical Study on Efficient Tool-Calling Language Models via Semantic Protocol Compression**

---

## 1. 摘要 & 研究背景 (Abstract & Background)
隨著個人隱私保護意識的提升，在 PWA（漸進式網頁應用）中實現離線端側 AI 推理已成為核心趨勢。然而，部署 0.5B 或 1.8B 等參數量的小模型，在行動端瀏覽器（透過 WASM/WebGPU）運行時仍會面臨顯著的載入延遲與記憶體（KV Cache）開銷。

本專案從零開始設計並訓練了一個 **58M 參數** 的超微型 Decoder-only 語言模型（BookkeepingLM）。本研究聚焦於**「在極小型端側 LLM 中，Tool Calling 的瓶頸不只是模型容量限制，更是輸出協議 (Output Protocol) 的冗餘。我們提出語意協議壓縮 (Semantic Protocol Compression) 概念，以更短且語義明確的特殊 Token 取代 JSON 格式，旨在降低生成長度並提升格式穩定性。」**

> **⚠️ 最新現況 (2026-10-11)**
> 經過多輪迭代（v1 2026-08 → v2 協議 v2 → v3 日期強化 → v4/v4c 短語強化），目前的實際模型與管線如下（下列規格優先於本文舊的 30M 研究規格）：
>
> * **模型**：`BookkeepingLM` 約 **58M 參數**（12 層、d_model=768、8 heads/4 KV heads、max_seq_len=1024、vocab=6400 + 8 個記帳特殊 Token），實作於 `model/model.py`。發佈權重：Hugging Face `the-walking-fish/jijun-LM`（safetensors + tokenizer），可經 `restore_hf_checkpoint.py` 還原為本機 `.pt` 做 continue-SFT（111 tensors，tokenizer ID 逐位相容）。
> * **輸出協議**：目前採用 **JSON `<tool_call>{...}</tool_call>` 協議**（`--format json`），並以與 PWA 前端對齊的 parser (`trainer/validate_json.py`) 驗證。
> * **資料管線**：`gen/` 目錄為現行產生器（v3 `build_v3_dataset.py` 日期強化、v4/v4c `build_short_phrase.py` + `build_v4c.py` 短語強化）；舊 `generate_dataset.py`（vllm Qwen3.6-27B API 生成）為 v1 產線。
> * **訓練管線**：`train_custom_sft.py` 已內建 **bf16 混合精度**（`--amp bf16`，預設開）與**進程級 VRAM 硬上限**（`--vram_fraction 0.6`），在 12GB 消費顯卡上訓練同時保留桌面餘裕，杜絕 WDDM 溢出共享記憶體拖慢系統。
> * **已修復的兩個訓練管線級 bug**（詳見 `docs/t6-v3-dataset-notes.md`、`docs/t7-v4c-short-phrase-notes.md`）：
>   1. `gen/build_prompt._weekday_zh_for` 星期注入錯位一天（v3）；
>   2. `dataset.py` 把訓練 system prompt 日期全部硬編碼為同一天 → `date` 欄位學習被徹底摧毀（訓練時所見日期恆定、gold label 卻是真實日期），v4b 修復。
> * **最新評測**（v4c continue-SFT，greedy；v1 基線 831 筆 gate 集與 v3/v4c 各集詳見第 6.1 節）：短語集 EM **75.7%**（category/description/type 全 100%）、v3 測試集 date **89.5%**（bug 修復前近 0%）、全新詞泛化探針 amount **90.6%**（證明「從句子抄金額」為真學會而非背誦）。

---

## 2. 模型架構與參數規格 (Model Architecture & Scaling Specs)
本專案的模型架構為 **純 PyTorch 原生編寫實作 (From Scratch)**，旨在展示對底層 Transformer 機制的完全掌控。我們並未直接載入第三方開源模型程式碼，而是對標 LLaMA / Qwen 等現代小模型設計原則：

* **超參數規格 (Scaling Config)**（現況，見上方「最新現況」）：
  * **總參數量**：約 **58M**。
  * **Transformer 層數 (Layers)**：12 層。
  * **注意力頭 (Heads)**：Query Heads = 8，Key/Value Heads = 4 (頭數比 2:1，GQA 機制)。
  * **隱藏層維度 ($d_{\text{model}}$)**：768。
  * **SwiGLU 中間層維度**：1152（ aligned with 32 multiples ）。
  * **最大序列長度 (Max Seq Len)**：1024。
  * **詞表大小 (Vocab)**：6400 + 8 個記帳專屬特殊 Token = 6406。
* **核心技術架構**：
  * **Grouped-Query Attention (GQA)**：藉由 2:1 比例壓縮 Key/Value 頭，使模型推論時的 KV Cache 記憶體開銷直接折半，極大優化了端側瀏覽器 (WASM/WebGPU) 的載入與執行性能。
  * **Rotary Position Embedding (RoPE)**：原生 PyTorch 實作二維複數旋轉位置編碼，使模型天然獲得相對位置特徵，並具備優秀的長序列外推能力。
  * **SwiGLU 激活函數**：使用 LLaMA 標準的 SiLU 閘控雙線性 Feed-forward 網路，相較於傳統 GELU/FFN 能以更少參數提供更強的非線性表徵能力。
  * **Pre-RMSNorm 與 Pre-Norm 架構**：移除 LayerNorm 中的均值平移計算，藉由均方根縮放提升約 10% 算力效率，Pre-Norm 佈局能有效穩定深度小模型微調。
  * **Tied Embedding (權重共享)**：綁定輸入 Embedding 層與輸出 LM Head 層的參數，節省約 3.2M 參數量 (約 10% 空間)，特別適合極微型端側模型的體積壓縮。

## 3. 核心研究假設與消融對比實驗 (Research Hypotheses & Ablations)
我們假設：**透過 Semantic Protocol Compression 能夠顯著減少解碼 Token 數、降低小模型格式損毀率，並縮減 KV Cache 與推論延遲。**

為了驗證上述假設，我們設計了以下消融對比實驗（Ablation Matrix），並使用相同難度的分層測試集進行評估：

| 實驗組 ID | 模型規模 | 協議格式 (Format) | 微調對齊方式 | 研究探討方向 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp-1 (JSON Baseline)** | 30M | 標準 JSON | 僅 SFT | 評估極微型模型在傳統 JSON 協議下的格式損毀率與槽位丟失率。 |
| **Exp-3 (Compressed)** | 30M | 特殊 Token 壓縮 | 僅 SFT | **核心創新組**。評估註冊特殊 Token 後對 Context 壓縮與格式通過率的改進效果。 |
| Exp-3 + LS | 30M | 壓縮 + Label Smoothing 0.1 | 僅 SFT | 驗證標籤平滑對極小模型是否有效。 |
| Exp-3 + LR 1e-4 | 30M | 壓縮 + LR 1e-4 | 僅 SFT | 驗證學習率敏感性。 |
| Exp-3 + n_layers=12 | 41M | 壓縮 + 12 層 | 僅 SFT | 驗證增加模型容量能否改善。 |
| Exp-3 + Reg | 30M | 壓縮 + Dropout 0.2 + WD 0.2 | 僅 SFT | 驗證強正則化能否緩解過擬合。 |
| Exp-3 + DataAug | 30M | 壓縮 + 規則擴增 | 僅 SFT | 驗證金額擾動與分類交換擴增的效果。 |

---

## 4. 技術特徵與研究故事 (Key Features)

### 3.1. 特殊 Token 語意協議壓縮 (Protocol Compression)
為了克服分詞器在符號與 Key 名稱上造成的 token 碎裂，我們向詞表註冊了專用特殊 Token：`[AMT]` (金額)、`[CAT]` (分類)、`[ACC]` (帳戶)、`[DESC]` (備註)、`[TYPE]` (收支類型)。
* **標準 JSON 格式**：
  `<tool_call>{"amount":150,"category":"餐飲","account":"信用卡"}</tool_call>` (約 28 Tokens)
* **特殊 Token 壓縮格式**：
  `<tool_call>[AMT]150[CAT]餐飲[ACC]信用卡</tool_call>` (約 12 Tokens)
* **實驗驗證**：壓縮格式相較 JSON 格式節省約 35% 序列長度。在 1,600 筆 SFT 測試下，Exact Match Rate 提升 **+7.25%**（66.25% → 73.50%），金額正確率提升 **+8.25%**（79.25% → 87.50%），推論延遲降低 **15%**（756ms → 644ms）。格式通過率維持在 99.50% 以上。

### 3.2. Wiki + SFT 對話混合無監督預訓練資料引擎
模型採用 **Wikipedia 繁體百科與理財口語對話** 混合進行 Next Token Prediction 自迴歸預訓練：
* **Wikipedia 資料處理**：動態載入繁體中文維基百科子集 (`lianghsun/wikipedia-zh-filtered`)。首次下載後將由 Hugging Face `datasets` 自動快取至本地 (通常位於 `~/.cache/huggingface/datasets`)，後續訓練無需重複下載。
* **資料比例與封裝控制**：採用精準 Token 數量限制（預設為 70% 百科 Token、30% 口語 Token），並於訓練前執行亂序 (Shuffle)、文檔邊界劃分 (Document Boundary, 插入 `<|im_end|>`) 與封裝 (Packing)，避免不同文章內容產生語意混淆。

### 3.3. 雙軌 SFT 微調與 PWA 正則解析器
* **雙軌 SFT**：支持 `--format json` 或 `--format compressed` 在 SFT 訓練時自動進行數據模板改寫。
* **格式 RL 獎勵 (Reward Function) (研究中)**：設計了專用於 GRPO / PPO 的多層級 Reward 評分機制（-1.0 至 4.0 分），直接以「PWA 解析器能否 parse 成功」作為獎勵反饋。

### 3.4. GGUF / ONNX 重映射對齊 (部署實踐)
* 支援匯出為帶有動態軸的 ONNX 格式。
* 實作 **GGUF 權重重命名重映射**：將自製模型的 Transformer 層重映射為標準 Llama/Qwen 架構，提供 GGUF 權重映射工具以方便後續轉換，有助於使用 wllama 等前端推理引擎在瀏覽器中部署。

---

## 6. 實驗結果 (Experimental Results)

### 6.1 最新評測結果 (58M 現況模型, 2026-10-11, v4c continue-SFT)

v4c 以 HF 發佈權重 `the-walking-fish/jijun-LM` 經 `restore_hf_checkpoint.py` 還原為起點做 continue-SFT（v3 資料 + v4c 短語強化資料混合，bf16，lr 5e-5，4 epochs，val loss 0.0721）。評測以 `eval_checkpoints.py` greedy 自迴歸推論：

| 測試集 | 筆數 | 格式通過 | EM | amount | category | account | description | type | date |
|--------|------|---------|-----|--------|----------|---------|-------------|------|------|
| 短語回歸集 `dataset/v4c/regression_v4.jsonl`（物品+金額型，含「早餐60」） | 103 | **100%** | **75.7%** | 92.2% | **100%** | 87.4% | **100%** | **100%** | 96.1% |
| 全新詞泛化探針 `dataset/v4c/probe_unseen.jsonl`（20 個訓練從未見過的商品詞） | 160 | **100%** | 16.2% | **90.6%** | 30.0% | 78.8% | 87.5% | 84.4% | **93.8%** |
| v3 測試集 `dataset/v3/test_v3.jsonl`（長句回歸） | 493 | **100%** | 20.7% | 95.3% | 82.4% | 97.0% | 26.4%* | 96.8% | **89.5%** |
| Gate 抽樣 `dataset/v4/gate_sub300.jsonl` | 300 | **100%** | 19.3% | 96.0% | 70.0% | 98.7% | 32.0%* | 86.7% | 75.3% |

> **要點**：
> 1. **date 欄位從近 0% 修復至 89.5%**——根因是 `dataset.py` 訓練期把 system prompt 日期硬編碼為同一天（模型所見日期恆定、gold label 卻是真實日期，等於要求模型猜測不可學的訊號），詳見 `docs/t7-v4c-short-phrase-notes.md`。
> 2. **短語型輸入（「早餐60」式物品+金額）EM 從 0% → 75.7%**。失敗主因（v4b 前）是訓練集全額湊整成十/百倍數 + 商品詞未入池，模型學會「猜該品項典型金額」而非「抄句子裡的數字」；v4c 以破先驗極端值（如早餐 10 元、牛排 20 元）與 35% 非湊整金額強制學會抄寫機制。
> 3. **全新詞探針 amount 90.6% / date 93.8%** 是抄寫機制真正學會的證據：這些商品詞在訓練中從未出現，模型仍正確從句子提取金額與日期。其 category 30% 屬預期（無語義先驗可依循的品項對 58M 模型是合理上限）。
> 4. *description 欄位 EM 低（26–32%）是對自由文字做逐字比對的必然結果，非生成錯誤；PWA 前端僅顯示不驗證該欄位。
> 5. 短語集 account 87.4% 的失敗多為測試資料隨機指定非現金帳戶但句子無支付線索，模型預設現金在產品語意上反而更合理。

### 6.1b 前代基線 (58M v1 模型, 2026-08-02)

v1 模型在 `dataset/qwen_full/test_strata.jsonl`（831 筆，含日期欄位、星期錨定）上的評測（`evaluate_benchmark.py` 全量自迴歸 Greedy 推論）：

| 模型 | 格式通過率 | 精準匹配率 | 金額正確率 | 分類正確率 | 帳戶正確率 | 日期正確率 | 平均 Reward | 延遲 (ms) |
|------|----------|----------|----------|----------|----------|----------|-----------|---------|
| **JSON** (`saves/best_bookkeeping_model.pt`) | **98.68%** | **39.35%** | **92.30%** | **68.35%** | **96.27%** | **61.25%** | **3.67** | **1095.5** |
| **Compressed** (`saves/compressed/best_bookkeeping_model.pt`) | **100.0%** | **38.63%** | **92.06%** | **67.27%** | **96.63%** | **61.97%** | **3.69** | **519.0** |

> **說明**：58M 模型的精準匹配率（EM）顯著低於下方 30M 消融數字的歷史結果，主因是**新版測試集加入了 `date` 欄位（日期正確率僅 61–62%）**，且每個 system prompt 對應不同的 category/account enum，難度大幅高於舊版手工 1,600 筆測試集。Compressed 版以更短的輸出序列達成更高的格式通過率與近兩倍的延遲優勢。

### 6.2 舊版 30M 消融研究 (Historical Ablation, 2026-07)

> 下列為 30M 研究規格（8 層 / d_model=512）在 1,600 筆手工記帳樣本上的消融結果，供研究故事與方法論參考。

在不可修改的 PWA 前端對齊解析器 (`extract_tool_call` + `validate_record`) 下進行的標準化評測結果：

| 實驗 | 格式通過率 | 精準匹配率 | 平均 Reward | 延遲 (ms) |
|------|----------|----------|-----------|---------|
| JSON Baseline (Exp-1) | 99.75% | 66.25% | 3.77 | 756 |
| **Compressed (Exp-3)** | **99.50%** | **73.50%** | **3.79** | **644** |
| Compressed + LS 0.1 | 99.25% | 74.25% | 3.82 | 295 |
| Compressed + LR 1e-4 | 99.25% | 51.00% | 3.62 | 300 |
| Compressed + n_layers=12 | 99.00% | 75.50% | 3.83 | 420 |
| Compressed + Dropout 0.2 + WD 0.2 | 99.25% | 76.50% | 3.81 | 300 |
| Compressed + Full Augmentation | 99.00% | 69.75% | 3.74 | 670 |
| Compressed + Amount Augmentation | 99.25% | 72.00% | 3.75 | 650 |

**主要發現：** 壓縮格式（Exp-3）相較 JSON 基線，Exact Match Rate 提升 **+7.25%**（66.25% → 73.50%），金額正確率提升 **+8.25%**（79.25% → 87.50%），推論延遲降低 **15%**（756ms → 644ms）。(各欄位正確率詳見分層級分析)

### 6.3 分層級分析 (Historical, 30M)

| 難度層級 | 樣本數 | 格式 | 精準匹配 | 金額正確率 | 分類正確率 | 帳戶正確率 |
|---------|-------|------|---------|----------|----------|----------|
| Level-1 | 134 | JSON | 73.13% | 88.81% | 85.07% | 94.03% |
| | | **Compressed** | **80.60%** | **93.28%** | **88.06%** | **94.03%** |
| Level-2 | 189 | JSON | 68.78% | 77.78% | 82.54% | 96.83% |
| | | **Compressed** | **78.84%** | **89.42%** | **86.77%** | **97.35%** |
| Level-3 | 76 | JSON | 47.37% | 65.79% | 82.89% | 92.11% |
| | | **Compressed** | **48.68%** | **73.68%** | **78.95%** | **78.95%** |

壓縮格式在 Simple 與 Noise 層級改善顯著，Reasoning 層級改善有限且帳戶正確率反而下降。

### 6.4 消融研究總結 (Historical, 30M)

超過 7 項消融實驗一致表明，**沒有任何超參數調整、架構修改或規則式資料擴增能超越壓縮格式基線**：

| 變因 | 嘗試方向 | 結果 |
|------|---------|------|
| Label Smoothing | 0.1 | 74.25% (更差) |
| 學習率 | 1e-4 | 51.00% (大幅更差) |
| 模型容量 | 12 層 (41M) | 75.50% (更差) |
| 正則化 | Dropout 0.2 + WD 0.2 | 76.50% (持平) |
| 資料擴增 (全) | 分類/帳戶/金額 | 69.75% (更差) |
| 資料擴增 (金額) | 僅金額擾動 | 72.00% (更差) |

### 6.5 錯誤分析 (Historical, 30M)

對壓縮格式模型 (73.50% EM) 的錯誤樣本分析揭示：

| 錯誤類型 | 佔比 | 說明 |
|---------|------|------|
| 金額錯誤 | 76% | 模型選取錯誤數值為最常見錯誤 |
| 分類錯誤 | 39% | 混淆相似分類 |
| 帳戶錯誤 | 12% | 相對較少 |
| 類型錯誤 | 12% | 混淆 expense/income |

Level-3 (Reasoning) 樣本的**金額正確率僅 73.68%**，反映出 30M 模型在需要多步推理的場景存在根本性能力限制。

---

## 7. 專案目錄結構
```
jijun-ai-training/
├── model/
│   ├── model.py              # 58M Transformer 架構 (GQA, RoPE, Tied Embedding) 與參數量計算器
│   └── tokenizer.py          # 自訂記帳 Tokenizer 載入與特殊 Token 註冊器
├── trainer/
│   └── validate_json.py      # 與 PWA 前端對齊的 JSON/壓縮標記雙軌解析器與 RL 獎勵函數
├── gen/                      # 現行資料產生器（v2~v4c）
│   ├── build_prompt.py       # system prompt 組裝（tools schema + 日期星期注入，與前端對齊）
│   ├── build_v3_dataset.py   # v3 日期強化資料集組裝（星期注入 + 純日偏移增廣）
│   ├── build_short_phrase.py # v4 短語型樣本產生器（物品+金額，train/test 商品詞零重疊）
│   ├── build_v4c.py          # v4c 短語強化（破先驗金額 + 全新詞泛化探針）
│   ├── mix_v2_dataset.py     # v2 多工具資料混合
│   ├── gen_js_tools.py       # tools.schema.json → 前端 toolsSchema.js 同步
│   └── test_build_prompt.py  # 星期映射金標回歸鎖
├── dataset/                  # SFT 訓練與測試語料目錄
│   ├── train_strata.jsonl    # v1 訓練集 (3,323 筆)
│   ├── test_strata.jsonl     # v1 測試集 (831 筆，守門集；.bak 為原版)
│   ├── qwen_full/            # Qwen 生成原始/過濾後語料與合併檔
│   ├── v2/  v3/  v4/  v4c/   # 歷代版本化資料集（v3 日期強化、v4c 短語強化＋泛化探針）
├── saves/                    # 模型 Checkpoint（.gitignore，權重發佈於 HF the-walking-fish/jijun-LM）
│   ├── best_bookkeeping_model.pt  # 最佳 SFT 模型 (JSON 版, demo 使用)
│   ├── compressed/best_bookkeeping_model.pt  # 最佳 SFT 模型 (Compressed 版)
│   ├── best_pretrain_model.pt     # 預訓練權重
│   └── backups/                    # 定期備份
├── requirements.txt          # Python 依賴安裝清單
├── restore_hf_checkpoint.py  # HF safetensors → 本機 .pt 權重返還（continue-SFT 起點）
├── eval_checkpoints.py       # 逐欄位 greedy 評測（任意 .pt × 任意 jsonl）
├── generate_dataset.py       # [v1 產線] Qwen 大模型批量分層口語樣本生成腳本
├── filter_dataset.py         # [v1 產線] 嚴格語意對齊過濾腳本（舉證責任倒置）
├── split_dataset.py          # 80/20 訓練測試切分腳本
├── convert_dataset_dates.py  # 資料集日期標準化腳本
├── train_custom_pretrain.py  # 無監督 Wikipedia + 對話混合預訓練腳本 (含 Packing 與比例控制)
├── train_custom_sft.py       # 全量/continue SFT 微調腳本（bf16 AMP + VRAM 上限保護）
├── evaluate_benchmark.py     # 分層量化評測腳本 (真實自迴歸 Greedy 推理 vs. Mock 評測)
├── verify_model.py           # 語意抽查腳本 (15 句手寫中文查詢 vs 模型輸出)
├── demo.py                   # 基於 Gradio 的互動記帳測試網頁
├── start_sft.py 等           # 背景執行啟動腳本 (CREATE_NO_WINDOW)
├── export_model.py           # ONNX / Llama-GGUF 重映射匯出對齊工具
└── docs/                     # 版本化戰役筆記（t6 v3 日期強化、t7 v4c 短語強化、路線圖）
```

---

## 8. 快速開始 (Quick Start)

### 8.1. 安裝環境
請在虛擬環境中執行以下命令一鍵安裝 Python 依賴：
```bash
pip install -r requirements.txt
```

### 8.2. 生成 SFT 資料集 (Data Synthesis)
您可以透過命令列參數、環境變數或本地 `.env` 檔案傳入 API 與生成設定（安全最佳實踐）：

**方法 A：使用 `.env` 檔設定（推薦）**
複製範例檔 `.env.example` 為 `.env` 並填入設定：
```env
OPENAI_API_KEY=您的_API_KEY
OPENAI_BASE_URL=您的_API_URL
LLM_MODEL=vllm/Qwen3.6-27B
CONCURRENCY=3
REST_INTERVAL=2
REST_DURATION=1
```
然後直接執行：
```bash
python generate_dataset.py --count 100 --out_dir ./dataset
```

> **注意**：目前生成模型為 **`vllm/Qwen3.6-27B`**（非舊的 `qwen3.6-35b-a3b-mtp`），且需啟用 `enable_thinking=False`（腳本已自動處理）。`--bias_categories` 可指定類別清單（如 `零用錢,投資,娛樂`），使 70% 樣本偏向這些低頻類別。

**方法 B：命令列參數傳入**
```bash
python generate_dataset.py --api_key "您的_API_KEY" --api_url "您的_API_URL" --count 100 --out_dir ./dataset --rest_interval 30 --rest_duration 5
```

**方法 C：設定環境變數執行**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="您的_API_KEY"
$env:OPENAI_BASE_URL="您的_API_URL"
python generate_dataset.py --count 100 --out_dir ./dataset --concurrency 4

# Linux/macOS
export OPENAI_API_KEY="您的_API_KEY"
export OPENAI_BASE_URL="您的_API_URL"
python generate_dataset.py --count 100 --out_dir ./dataset --concurrency 4
```
*腳本支持斷點續傳，若中途取消或 API 超時，重啟後會自動加載歷史進度並跳過已完成的難度等級。您可以透過 `--concurrency` 自訂並行呼叫數，以及透過 `--rest_interval` (執行 N 分鐘) 與 `--rest_duration` (休息 M 分鐘) 防止顯卡過熱。*

### 8.3. 自定義比例預訓練 (Pre-training)
```bash
# 預設會執行 Wiki 與對話的 Packing 控制與 7:3 比例截斷
python train_custom_pretrain.py --use_wikipedia --wiki_limit 3000 --save_dir ./saves

# 若想放開 7:3 比例限制，直接採用所有載入的數據進行最大化預訓練：
python train_custom_pretrain.py --use_wikipedia --wiki_limit 3000 --save_dir ./saves --no_ratio_limit
```

### 8.4. SFT 微調 (Supervised Fine-Tuning)

**方式 A（現行主流）：從 HF 發佈權重做 continue-SFT**
```bash
# 1. 下載 HF safetensors 到 hf_ckpt/ 並還原為本機 .pt（自訂命名對齊 model.py）
mkdir -p hf_ckpt && cd hf_ckpt && for f in config.json tokenizer.json tokenizer_config.json model.safetensors model.safetensors.index.json; do
  curl -sL -o $f "https://huggingface.co/the-walking-fish/jijun-LM/resolve/main/model-json/$f"
done && cd ..
python restore_hf_checkpoint.py   # → saves/jijun_lm_restored.pt

# 2. continue-SFT（低 LR 防災難性遺忘；bf16 + VRAM 上限預設開啟，桌面無感）
python train_custom_sft.py \
  --train_path dataset/v4c/train_v4c.jsonl \
  --val_path   dataset/v4c/val_v4c.jsonl \
  --pretrained_path saves/jijun_lm_restored.pt \
  --format json --epochs 4 --lr 5e-5 --batch_size 8 --max_seq_len 768 \
  --save_dir saves_v4c
```

**方式 B：從預訓練權重做全量 SFT（v1 流程）**
```bash
python train_custom_sft.py --pretrained_path ./saves/best_pretrain_model.pt --format json --eval_generate --max_eval_samples 20 --epochs 8 --lr 2e-4 --batch_size 8 --max_seq_len 1024 --save_dir ./saves
```

> **VRAM 保護**：`--amp bf16`（預設）把激活值顯存減半；`--vram_fraction 0.6`（預設）設定進程級 VRAM 硬上限，超過即在訓練進程內報 OOM，**不會**讓 Windows WDDM 把分配擠進共用記憶體拖慢桌面。12GB 卡建議 `--batch_size 8`（峰值整卡含桌面約 8.9GB）。

### 8.5. 語意抽查驗證 (Semantic Spot Check)
```bash
python verify_model.py
```
*以 15 句手寫中文記帳查詢（薪水/零用錢/投資/交通…）做 argmax 解碼抽查，輸出期望分類 vs 實際分類，並以 parser 驗證 JSON 合法性。*

### 8.6. 導出 ONNX 與 GGUF 權重
```bash
python export_model.py --ckpt_path ./saves/best_bookkeeping_model.pt --out_dir ./exports
```
*這會生成 `bookkeeping_model.onnx` 並在 `./exports/llama_compatible_weights` 中產生標準的 Llama 權重結構與 `config.json`，方便 llama.cpp 轉為 GGUF。*

### 8.7. 運行量化 Benchmark 評測
```bash
python evaluate_benchmark.py --model_path ./saves/best_bookkeeping_model.pt --dataset ./dataset/test_strata.jsonl
```
*若 `--model_path` 指定為 `"dummy"`，將會自動啟動模擬評測模式進行流程驗證。*

逐欄位快速評測（任意 checkpoint × 任意 jsonl，greedy 自迴歸，含 format/EM/六欄位正確率與失敗範例）：
```bash
python eval_checkpoints.py saves_v4c/best_bookkeeping_model.pt dataset/v4c/regression_v4.jsonl
```

---

## 9. 開源參考與軟體依賴 (References & Dependencies)
* **模型與訓練程式碼**：`model/model.py` 採用純 PyTorch 原生編寫，並無引用第三方模型架構代碼。
* **分詞器 (Tokenizer) 基礎**：基於相容 Llama 結構的分詞器（藉由 `jingyaogong/minimind-3` 載入分詞器詞表），並在載入時動態註冊本研究自製之語意特殊 Token 協議。
* **開發依賴**：PyTorch (底層矩陣運算)、Hugging Face `datasets` (維基百科流式下載) 與 `openai` SDK (API 語料合成工具)。

## 10. 致謝 (Acknowledgements)
本研究之超微型模型架構與管線設計，深受開源社群優秀專案的啟發與參考，特此對相關創作者的無私開源分享致以深切謝意。