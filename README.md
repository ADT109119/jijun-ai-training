# 基於語意協議壓縮的 30M 參數以下高效端側工具調用語義模型研究
> **An Empirical Study on Efficient Tool-Calling Language Models under 30M Parameters via Semantic Protocol Compression**

---

## 1. 摘要 & 研究背景 (Abstract & Background)
隨著個人隱私保護意識的提升，在 PWA（漸進式網頁應用）中實現離線端側 AI 推理已成為核心趨勢。然而，部署 0.5B 或 1.8B 等參數量的小模型，在行動端瀏覽器（透過 WASM/WebGPU）運行時仍會面臨顯著的載入延遲與記憶體（KV Cache）開銷。

本專案從零開始設計並訓練了一個 **30M 參數** 的超微型 Decoder-only 語言模型（ToolMind-30M）。本研究聚焦於**「在極小型端側 LLM（<30M）中，Tool Calling 的瓶頸不只是模型容量限制，更是輸出協議 (Output Protocol) 的冗餘。我們提出語意協議壓縮 (Semantic Protocol Compression) 概念，以更短且語義明確的特殊 Token 取代 JSON 格式，旨在降低生成長度並提升格式穩定性。」**

---

## 2. 模型架構與參數規格 (Model Architecture & Scaling Specs)
本專案的模型架構為 **純 PyTorch 原生編寫實作 (From Scratch)**，旨在展示對底層 Transformer 機制的完全掌控。我們並未直接載入第三方開源模型程式碼，而是對標 LLaMA / Qwen 等現代小模型設計原則：

* **超參數規格 (Scaling Config)**：
  * **總參數量**：約 **30M**。
  * **Transformer 層數 (Layers)**：8 層。
  * **注意力頭 (Heads)**：Query Heads = 8，Key/Value Heads = 4 (頭數比 2:1，GQA 機制)。
  * **隱藏層維度 ($d_{\text{model}}$)**：512。
  * **SwiGLU 中間層維度**：1536（ aligned with 32 multiples ）。
  * **最大序列長度 (Max Seq Len)**：512。
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
| **Exp-0 (Direct ChatML)** | 30M | 無 (純口語對話) | 僅 SFT | **基準對照組**。不使用任何 Tool Schema 或格式語法，直接輸出一般口語理財文字，用以評估特定格式協議本身是否會對模型提取能力造成額外開銷。 |
| **Exp-1 (JSON Baseline)** | 30M | 標準 JSON | 僅 SFT | 評估極微型模型在傳統 JSON 協議下的格式損毀率與槽位丟失率。 |
| **Exp-2** | 30M | 標準 JSON | SFT + RL (研究中) | 驗證強化學習（RL）能否糾正小模型在 JSON 格式輸出上的幻覺與括號閉合問題。 |
| **Exp-3 (Compressed)** | 30M | 特殊 Token 壓縮 | 僅 SFT | **核心創新組**。評估註冊特殊 Token 後對 Context 壓縮與格式通過率的改進效果。 |
| **Exp-4 (Optimal)** | 30M | 特殊 Token 壓縮 | SFT + RL (研究中) | 驗證特殊 Token 協議與格式 RL 獎勵相結合，能否達到最佳端側性能。 |

---

## 3. 技術特徵與研究故事 (Key Features)

### 3.1. 特殊 Token 語意協議壓縮 (Protocol Compression)
為了克服分詞器在符號與 Key 名稱上造成的 token 碎裂，我們向詞表註冊了專用特殊 Token：`[AMT]` (金額)、`[CAT]` (分類)、`[ACC]` (帳戶)、`[DESC]` (備註)、`[TYPE]` (收支類型)。
* **標準 JSON 格式**：
  `<tool_call>{"amount":150,"category":"餐飲","account":"信用卡"}</tool_call>` (約 28 Tokens)
* **特殊 Token 壓縮格式**：
  `<tool_call>[AMT]150[CAT]餐飲[ACC]信用卡</tool_call>` (約 12 Tokens)
* **研究假設**：預估能節省高達 50%+ 的 Token 生成開銷。透過消除冗餘大括號與屬性名稱，可大幅縮減 Prefill 與 Decode 階段的計算成本，並預期能降低小模型大括號未閉合導致的格式損毀率。

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

## 4. 初步實驗結果 (Preliminary Results)
在相同驗證語料下，對不同實驗組別進行了初步測試，結果如下表所示（數據為特定實驗參數下之初步測試值，僅作學術對比參考）：

| 實驗組 ID | 協議格式 (Format) | 格式通過率 (Format Pass) | 欄位提取準確率 (Slot F1) | 平均生成 Token 數 | 平均推論延遲 (Latency) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp-0 (Direct ChatML)** | 純口語對話 | N/A (不適用) | N/A (不適用) | — | ~15ms |
| **Exp-1 (JSON Baseline)** | 標準 JSON | 82.3% | 88.1% | ~28 Tokens | ~42ms |
| **Exp-3 (Compressed SFT)**| 特殊 Token | 96.5% | 94.2% | ~12 Tokens | ~23ms |

*實驗結果顯示，特殊 Token 語意協議能以更少的解碼字數換取更高的格式穩定性，初步驗證了輸出協議壓縮在極小模型上的可行性。*

---

## 5. 專案目錄結構
```
jijun-ai-training/
├── model/
│   ├── model.py              # 30M Transformer 架構 (GQA, RoPE, Tied Embedding) 與參數量計算器
│   └── tokenizer.py          # 自訂記帳 Tokenizer 載入與特殊 Token 註冊器
├── trainer/
│   └── validate_json.py      # 與 PWA 前端對齊的 JSON/壓縮標記雙軌解析器與 RL 獎勵函數
├── dataset/                  # SFT 訓練與測試語料目錄
├── requirements.txt          # Python 依賴安裝清單
├── generate_dataset.py       # 大模型批量分層口語樣本生成腳本 (支援斷點續傳與速率限制)
├── train_custom_pretrain.py  # 無監督 Wikipedia + 對話混合預訓練腳本 (含 Packing 與比例控制)
├── train_custom_sft.py       # 雙軌全量 SFT 微調腳本 (含 CPU 驗證防卡死優化)
├── evaluate_benchmark.py     # 分層量化評測腳本 (真實自迴歸 Greedy 推理 vs. Mock 評測)
└── export_model.py           # ONNX / Llama-GGUF 重映射匯出對齊工具
```

---

## 6. 快速開始 (Quick Start)

### 6.1. 安裝環境
請在虛擬環境中執行以下命令一鍵安裝 Python 依賴：
```bash
pip install -r requirements.txt
```

### 6.2. 生成 SFT 資料集 (Data Synthesis)
您可以透過命令列參數直接傳入，或設定環境變數以防金鑰寫死（安全最佳實踐）：

**方法 A：命令列參數傳入**
```bash
python generate_dataset.py --api_key "您的_API_KEY" --api_url "您的_API_URL" --count 100 --out_dir ./dataset
```

**方法 B：設定環境變數執行**
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
*腳本支持斷點續傳，若中途取消或 API 超時，重啟後會自動加載歷史進度並跳過已完成的難度等級。您可以透過 `--concurrency` 自訂並行呼叫數以控制生成速度（預設為 2）。*

### 6.3. 自定義比例預訓練 (Pre-training)
```bash
# 預設會執行 Wiki 與對話的 Packing 控制與 7:3 比例截斷
python train_custom_pretrain.py --use_wikipedia --wiki_limit 3000 --save_dir ./saves

# 若想放開 7:3 比例限制，直接採用所有載入的數據進行最大化預訓練：
python train_custom_pretrain.py --use_wikipedia --wiki_limit 3000 --save_dir ./saves --no_ratio_limit
```

### 6.4. 全量 SFT 微調 (Supervised Fine-Tuning)
你可以透過切換 `--format` 來測試不同協議的消融結果：
```bash
# 進行特殊 Token 壓縮協議的微調，並設定 SFT 學習率（預設為 2e-4）
python train_custom_sft.py --format compressed --eval_generate --max_eval_samples 5 --epochs 8 --lr 2e-4
```

### 6.5. 導出 ONNX 與 GGUF 權重
```bash
python export_model.py --ckpt_path ./saves/best_bookkeeping_model.pt --out_dir ./exports
```
*這會生成 `minimind_bookkeeping.onnx` 並在 `./exports/llama_compatible_weights` 中產生標準的 Llama 權重結構與 `config.json`，方便 llama.cpp 轉為 GGUF。*

### 6.6. 運行量化 Benchmark 評測
```bash
python evaluate_benchmark.py --model_path ./saves/best_bookkeeping_model.pt --dataset ./dataset/test_strata.jsonl
```
*若 `--model_path` 指定為 `"dummy"`，將會自動啟動模擬評測模式進行流程驗證。*

---

## 7. 開源參考與軟體依賴 (References & Dependencies)
* **模型與訓練程式碼**：[model.py](file:///c:/Users/me/OneDrive/桌面/HTML/輕鬆記帳/tools/jijun-ai-training/model/model.py) 採用純 PyTorch 原生編寫，並無引用第三方模型架構代碼。
* **分詞器 (Tokenizer) 基礎**：基於相容 Llama 結構的分詞器（藉由 `jingyaogong/minimind-3` 載入），並在載入時動態註冊本研究自製之語意特殊 Token 協議。
* **開發依賴**：PyTorch (底層矩陣運算)、Hugging Face `datasets` (維基百科流式下載) 與 `openai` SDK (API 語料合成工具)。

## 8. 致謝 (Acknowledgements)
本研究之超微型模型架構與管線設計，深受開源社群優秀專案 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的啟發與參考，特此對該專案創作者的無私開源分享致以深切謝意。