# 學術研究與研究所推甄作品集規劃書 (Academic Portfolio Blueprint)

本規劃書彙整了技術路線，旨在為您的子模組專案建立一個**高辨識度、結構嚴謹、具備學術論文厚度**的推甄代表作。

---

## 1. 論文 / 技術報告題目推薦
> **《Efficient Tool-Calling Language Models under 30M Parameters via Semantic Protocol Compression》**
> *(基於語意協議壓縮的 30M 參數以下高效能工具調用語言模型研究)*

---

## 2. 核心架構與參數規格 (Scaling Config)
為實現極限端側優化（瀏覽器 WASM / WebGPU 部署），模型規格設定如下：

* **模型參數量**：約 **30M**
* **Hidden Dimension ($d_{\text{model}}$)**：`512`
* **Attention Heads ($n_{\text{heads}}$)**：`8`
* **Key/Value Heads ($n_{\text{kv\_heads}}$)**：`4` (GQA 比例 2:1)
* **Transformer Layers**：`8`
* **SwiGLU 中間維度**：`1536`
* **最大序列長度 (Max Sequence Length)**：`512`
* **詞表 (Tokenizer)**：以 MiniMind/Qwen 的 6,400 詞表為基礎，**註冊專用特殊 Token** (`[AMT]`, `[CAT]`, `[ACC]`, `[DESC]`, `[TYPE]`)。
* **Tied Embedding**：輸入 Embedding 與輸出 LM Head 權重綁定，節省約 35% 參數量並加快收斂。

---

## 3. 數據工程與預訓練語料 (Pretrain Data Engine)
模型將採用 **70% 百科 + 30% 口語對話** 的混合數據集進行無監督自迴歸預訓練：

```
                      +------------------------------------+
                      |     預訓練混和語料 (Pretrain)       |
                      +------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
  +-------------------------------+             +-------------------------------+
  |  70% Wikipedia 繁體中文語料    |             |  30% 日常生活與消費對話語料   |
  | - 透過 datasets 線上載入      |             | - 透過自建 API 批量合成       |
  | - 首次執行後會自動本地快取    |             | - 模擬日常理財與記帳問答      |
  +-------------------------------+             +-------------------------------+
```

* **Wikipedia 處理**：使用 `datasets` 庫下載並載入繁體中文維基百科子集 (`lianghsun/wikipedia-zh-filtered`)，首次下載將自動進行本地快取。這展示了您對 NLP 資料處理與快取優化管線的控制。
* **生活理財語料**：利用您的自建 LLM API 批量生成 10MB–20MB 的日常閒聊與消費對話，增強模型對非結構化口語的適應力。

---

## 4. 工具調用協議設計 (Tool-Calling Protocol)
本研究的核心亮點在於**「拋棄冗長 JSON，採用專用語意 Token 壓縮」**：

* **傳統 JSON 格式**：
  `<tool_call>{"amount":150,"category":"餐飲","account":"信用卡"}</tool_call>` (28 Tokens)
* **專用標記協議**：
  `<tool_call>[AMT]150[CAT]餐飲[ACC]信用卡</tool_call>` (12 Tokens)
* **學術故事**：
  透過向詞表註冊 `[AMT]` 等專用語意標記，消除 BPE 分詞器在符號與 Key 名稱上的碎裂化，**節省高達 57% 的 Token 生成長度**，顯著降低 KV Cache 記憶體佔用與端側推論延遲。

---

## 5. 評測基準與指標 (Benchmark & Ablation)
評測腳本 [evaluate_benchmark.py](file:///c:/Users/me/OneDrive/桌面/HTML/輕鬆記帳/tools/jijun-ai-training/evaluate_benchmark.py) 將量化評估以下指標：

1. **Token 壓縮與推理效率 (Inference Efficiency)**：
   * **Token 節省率 (%)**：對比 JSON 協定所節省的 Token 比例。
   * **推論延遲 (Latency)**：計算 prefill 與 decode 階段的平均毫秒數。
   * **KV Cache 記憶體佔用**。
2. **動態 Schema 抗幻覺率 (Dynamic Schema Anti-Hallucination Rate)**：
   * 在 System Prompt 中隨機變更可用的分類/帳戶 Enum 列表，統計模型輸出的 `[CAT]` 與 `[ACC]` 是否 100% 侷限在給定清單中，有無產生幻覺分類。
3. **難度退化曲線 (Difficulty Degradation Curve)**：
   * 對比模型在 Level-1 (直白無噪聲)、Level-2 (日常修飾與雜訊) 與 Level-3 (推理與時間計算) 上的槽位提取準確率。

---

## 6. 技術報告 (Technical Report) 結構大綱

在您未來撰寫推甄技術論文或 GitHub README 時，建議採取以下標準學術結構：

```markdown
# 論文結構建議 (Paper / Technical Report Outline)

1. **Abstract (摘要)**
   - 研究動機：端側離線 LLM 算力瓶頸。
   - 提出方案：在 30M 規模下，利用專用語意 Token 壓縮，優化記帳窄域的 Tool Use 表現。
   - 核心實驗結果：節省 57% Token，抗幻覺率達 XX%。

2. **Introduction (引言 & 背景)**
   - 隱私保護與 PWA 離線 AI。
   - 超小量級模型（<60M）進行 Function Calling 的物理局限性。

3. **Methodology (研究方法)**
   - 30M 模型結構配置（GQA, SwiGLU, RoPE）。
   - 數據引擎設計：Wikipedia 載入快取 + API 對話合成。
   - 協議壓縮設計：專用語意 Token 設計。

4. **Experimental Setup (實驗設定)**
   - 數據集劃分與消融實驗矩陣設計。
   - 評測指標 (Benchmark Metrics) 的數學定義。

5. **Results & Analysis (結果與消融分析)**
   - 實驗一：Token 效率與 Latency 對比。
   - 實驗二：抗幻覺率對比。
   - 實驗三：SFT 收斂速度對比（檢視 Loss 曲線）。

6. **Limitations & Future Work (局限性與展望)**
   - 探討小模型在多步邏輯推理 (Multi-step ReAct) 上的能力上限。

7. **Conclusion (結論)**
```
