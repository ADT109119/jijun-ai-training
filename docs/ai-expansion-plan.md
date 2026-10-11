# BookkeepingLM 功能擴充規劃書 — 從「單一記帳解析」到「多工具語意助理」

> 撰寫：2026-08-20（雙模型審核後之規劃）
> 現況基線：58M 模型（12 層 / d_model=768 / GQA 8:4 / seq 1024 / vocab 6406）
> 現況指標（831 筆測試）：JSON 格式 98.68% / EM 39.35% / 金額 92.30% / 分類 68.35% / 帳戶 96.27% / **日期 61.25%（最弱）**
> 現況資料：4,154 筆（train 3,323 / test 831），全部為單一工具 `add_record`
> 主專案現況：`src/js/aiService.js`（wllama WASM + GGUF q4_0 34MB + 動態 enum prompt + 雙軌解析 + 規則降級）

> **📌 進度更新（2026-10-11）**
> * **T6 日期強化：✅ 完成**（v3 資料 + `dataset.py` date bug 修復，實測 date **89.5%**，超前 75%+ 目標；詳見 `docs/t6-v3-dataset-notes.md`、`docs/t7-v4c-short-phrase-notes.md`）。
> * **T4.5（路線圖外插入）短語型輸入強化：✅ 完成**（v4c，「早餐60」型 EM 0%→75.7%，泛化探針 amount 90.6%；評測詳見 README 6.1）。
> * T1-T3（transfer/debt）：資料與 Protocol v2 規格就緒（`docs/p0-protocol-v2-spec.md`、`gen/mix_v2_dataset.py`），**尚未跑 Stage A continue-SFT**——下一梯次。
> * 訓練基線更新：continue-SFT 一律 `--lr 5e-5`、起點改用 HF 權重返還（`restore_hf_checkpoint.py`）、`--amp bf16 --vram_fraction 0.6 --batch_size 8`（見 README 8.4）；本文件第 4 節 `lr 2e-4 / epochs 8` 為 v1 全量 SFT 舊參數，僅適用從 pretrain 重訓。

---

## 0. 一句話方向

**模型永遠不做計算，只做「自然語言 → 結構化意圖」。** 58M 的容量已逼近上限（消融實驗證明加層/擴增都更差），所以擴充路線不是「讓模型更聰明」，而是：

1. 把單一 `add_record` 工具擴充為**多工具 tool-calling**（模型選工具 + 填槽位）
2. 所有計算（統計、預算、餘額、分期）留在 app 端，模型只產出查詢意圖讓 app 執行
3. 保護 `add_record` 既有基線（守門指標），逐步加入新工具

這同時強化了研究的學術故事：論文標題就是 "Tool-Calling"，但目前只有 1 個 tool——擴充成多工具、多次呼叫、按 UI 情境注入工具，才是真正的 tool-calling 研究。

---

## 1. 核心架構建議：單一工具 Schema 合約（最重要的一項）

目前「工具定義」存在三個地方且靠人肉同步：
- 訓練資料的 system prompt（generate_dataset.py 內嵌）
- 前端 system prompt（aiService.js `generateSystemPrompt` 內嵌）
- 前端解析器（`extractToolCall` 的欄位假設）

**建議建立 `tools.schema.json` 作為單一真相源**，由它生成：
1. 訓練資料的 system prompt（Qwen 生成時的 tool 定義）
2. 前端 system prompt
3. 前端 validator（解析後比對 enum、型別、必填欄位）

任何擴充動 schema 只改一個檔，三端自動對齊。這是後續所有 phase 的地基，先做。

```
tools.schema.json（單一真相源）
  ├── 每個 tool: { name, description, parameters(JSON Schema), 欄位 enum 來源（動態/靜態） }
  ├── 動態 enum 標記: category→帳本分類、account→帳本帳戶、contact→聯絡人（運行時注入）
  └── 生成器:
      gen/system_prompt.py    → 訓練用 prompt（含日期錨定）
      js/toolsSchema.js       → 前端 import（prompt 組裝 + validator）
```

---

## 2. 功能候選清單（依 價值 × 58M 可行性 排序）

### Tier 1 — 純填槽、零計算（58M 最吃得消，先做）

| ID | Tool | 範例輸入 | 輸出 args | 難度 |
|----|------|---------|-----------|------|
| T1 | `transfer` 轉帳 | 「從現金轉 500 到信用卡」 | `{from_account, to_account, amount, date}` | 低（account 動態 enum 已驗證可行） |
| T2 | `add_debt` 記欠款 | 「借小明 500 元」 | `{contact, amount, direction: lend/borrow, date}` | 低（contact 動態 enum 仿 account） |
| T3 | `settle_debt` 還欠款 | 「小明還了我 200」 | `{contact, amount, date}` | 低 |
| T4 | `add_recurring` 週期交易 | 「每月 5 號繳房租 12000」 | `{amount, category, account, schedule: monthly, schedule_value: 5, date}` | 中（schedule enum 要設計好） |

### Tier 2 — 格式升級（不改工具、改輸出結構）

| ID | 能力 | 範例 | 風險 |
|----|------|------|------|
| T5 | **多筆紀錄批次解析** | 「昨天午餐 200，晚上聚餐 1200」→ 2 個 `add_record` tool_call | 中（58M 輸出 list 的格式穩定性；max_tokens 128→384） |
| T6 | 日期強化（先於 T5 做） | 現有 date 正確率僅 61%，新工具多數帶日期/排程，**先把日期槽修起來再擴** | 低（純資料面：加多日期表達式樣本） |

### Tier 3 — 查詢規劃器（模型→查詢，app→計算，價值最高）

| ID | Tool | 範例輸入 | 輸出 args | app 端執行 |
|----|------|---------|-----------|-----------|
| T7 | `query_stats` | 「這個月餐飲花了多少」 | `{metric: total, dimension: category, value: 餐飲, period: this_month}` | statistics.js 現有計算 → UI 顯示 |
| T8 | `query_budget` | 「預算超了嗎」 | `{dimension: category, period: this_month}` | budgetManager 比較 → 顯示 |
| T9 | `query_balance` | 「信用卡帳單日還有幾號」 | `{account: 玉山信用卡}` | dataService 讀取 → 顯示 |

> 關鍵：period 用 enum（`this_week/this_month/last_month/this_year`）不要讓模型算日期——模型只選 enum，app 用 `getCurrentDateAnchor` 同一套邏輯解析。
> T7 把「報表頁」變成可用口語查詢的入口，是對產品價值最大的一步。

### Tier 4 — 進階（視 Tier 1-3 成果再決定）

| ID | 能力 | 說明 | 風險 |
|----|------|------|------|
| T10 | 分期掛接 | 「這筆刷卡分 12 期」→ 需引用最近一筆紀錄，引入「上下文引用」概念 | 高（需要把用戶紀錄塞 prompt，1024 ctx 很吃緊） |
| T11 | 多輪修正 | 「不是 1200，是 1500」→ `modify_record` | 中（ChatML 已支援多輪，需新資料格式） |
| T12 | 情境感知分類建議 | system prompt 附最近 5 筆紀錄，模型做重複偵測/分類修正 | 高（ctx 預算 + 資料量需求大） |
| T13 | RL 格式獎勵 | README 已設計的 -1~4 reward 正式上 GRPO | 研究導向 |

---

## 3. 資料管線調整

### 3.1 generate_dataset.py 泛化
現況：硬編 add_record 的 15 分類 × 3 難度。改為：
```
--task transfer|debt|recurring|multi|query|...
每個 task = { tool schema (來自 tools.schema.json), 口語 pattern 庫, 難度層級定義, 動態 enum 池 }
```
沿用 Qwen3.6-27B（enable_thinking=False）+ 斷點續傳 + `--bias_categories` 偏斜機制。

### 3.2 各 tool 樣本量預算
參考現況密度（4,154 筆撐 1 個 tool × 15 分類 × 3 難度）：
- **每個新 tool：raw 1,500~2,500 筆 → 過濾後 ~1,200-1,800 筆**（少於 1,000 筆乾淨樣本，58M 槽位正確率會崩）
- query_stats 建議 2,000+（period × metric × dimension 組合多）

### 3.3 遺忘防護（比樣本量更重要）
消融實驗已證明此模型對資料配比極敏感（擴增全部更差）。規則：
1. **每次 SFT 的混合資料 = 40-50% 既有 add_record 資料 + 50-60% 新 tool 資料**（新 tool 分階段加，一次加 1-2 個）
2. 保留 `test_strata.jsonl` 原文不動，**永久作為 add_record 守門測試集**
3. 新 tool 各自的 test set 不混入守門集

### 3.4 過濾
沿用 filter_dataset.py「舉證責任倒置」思路，但每 tool 加一層：**schema 硬驗證**（args 必須符合 tools.schema.json，enum 必須在動態池內），這層可以直接用 generateSystemPrompt 的 enum 池生成，零人工。

### 3.5 日期強化（T6，Phase 1 前置）
在現有的「今天是 YYYY-MM-DD（星期X）」錨定上，補日期表達式覆蓋度：
- 相對日（昨天/前天/大前天/上星期幾/月初/月末）
- 絕對日（1月5號/1/5/去年12月）
- 未來日（明天/下週一）— 週期交易需要
目標：date 正確率 61% → 75%+ 再開新 tool。

---

## 4. 訓練策略

1. **不從 pretrain 重訓**：以 `saves/best_bookkeeping_model.pt`（現 58M JSON 版）為起點續 SFT。每階段的 checkpoint 另存（`saves/v2_transfer/`、`saves/v2_debt/`…），隨時可回退。
2. **分階段 rollout**：
   - Stage A：date 強化 + transfer + debt（一次兩個低風險 tool）
   - Stage B：recurring + 多筆批次（T5）
   - Stage C：query 系（T7-T9）
   - 每階段：混合 SFT → per-task 評測 → 守門指標通過才進入下階段
3. **超參**：維持現況 `--lr 2e-4 --epochs 8 --batch_size 8 --max_seq_len 1024`（消融證明動超參只會更差）；`max_tokens` 推論端 128→384（多筆批次需要）。
4. **Compressed track**：研究用途維持雙軌，但產品線只跑 JSON（格式 98.68% 已夠，compressed 的 +1.32pp 不值得維護兩套前端解析）。
5. **量化**：每階段 SFT 完成 → `export_all_quants.py` 出 q4_0/q5_0/q8_0 → 上傳 HF（`the-walking-fish/jijun-LM-GGUF`）→ aiService 的 ETag 更新機制會自動讓用戶端看到新版本（現成機制，不用改）。

---

## 5. 主專案調整（jijun）

### 5.1 aiService.js 重構
| 現況 | 改動 |
|------|------|
| `generateSystemPrompt(categories, accounts)` 內嵌單一 toolDef | `generateSystemPrompt(ctx)` 讀 `toolsSchema.js`，依 **UI 情境** 注入相關 tools（記帳頁：add_record/transfer/debt/recurring；報表頁：query 系；避免 58M 在長 prompt 下迷失） |
| `extractToolCall(text)` 回傳單一物件 | `parseToolCalls(text)` 回傳 **陣列**（`/<tool_call>[\s\S]*?<\/tool_call>/g` 全域匹配），每個附 tool name |
| 無 validator | 每 tool 一個 validate：enum 比對（category/account/contact 必須在用戶清單，否則標記 `needsConfirmation`）、型別、必填 |
| `_ruleInference` 降級 | 維持 add_record 專用；其他 tool 降級時**不嘗試**（UI 顯示「此指令需要 AI 模型」） |
| `max_tokens: 128` | → 384 |
| 聯絡人清單注入 | 仿 account enum：`getContacts()` 取名稱；超過 12 個時截斷（模型記不住長 enum）+ validator 端模糊匹配兜底 |

### 5.2 執行層（新 `aiActionExecutor.js` 或放 aiService）
```
add_record    → dataService.addRecord
transfer      → 現有轉帳建立（雙腿紀錄，同 processAmortizations 的 transfer 對邏輯）
add_debt      → dataService.addDebt
settle_debt   → dataService.settleDebt（走 resolveDefaultSettleAccountId，還款不入卡規則自動適用）
add_recurring → dataService.addRecurringTransaction
query_stats   → statistics.js → 回傳結果物件給 UI 渲染（不建紀錄）
```
**query 系與 action 系分流**：action 需要用戶確認才寫入；query 直接執行顯示。

### 5.3 UI 流程（關鍵體驗設計）
```
AI 輸入列 → parseToolCalls → 預覽卡片（1..N 個動作）
  每個卡片: [icon] tool 名稱 + 欄位摘要 + needsConfirmation 標記
  [全部確認] [逐一編輯] [取消]
確認後 → executor 寫入 → 既有 toast/導覽
```
- **多筆動作絕不自動執行**，一律過預覽卡（58M 格式 98.68% 意味著 1.3% 會錯，批量寫入錯誤成本太高）
- query 結果用現有卡片樣式，不新增 UI 系統
- AI 入口：記帳頁（現有）+ 報表頁（T7 上線後加入口）

### 5.4 測試策略（接軌 jijun 鐵律）
- 每個新 tool 的 validator：unit test（mock 模型輸出，含邊界：enum 外、缺欄位、多筆、思考鏈標籤）
- executor 各 tool：mock dataService 的 unit test（transfer 雙腿、settle_debt 走 FIFO 沖銷）
- E2E（Playwright）：至少一條「口語→預覽→確認→帳上有紀錄」完整路徑
- 守門：每次新模型上傳後，demo 頁 15 句手寫抽查（verify_model.py 的 15 句 + 新 tool 各 10 句）

### 5.5 FEATURES.md
排入待開發：`#AI2 AI 多工具擴充（v2 協議 + transfer/debt/recurring + NL 查詢）`，分 Tier 追蹤。

---

## 6. 路線圖

| Phase | 內容 | 產出 | 估時（含訓練等待） |
|-------|------|------|------|
| **P0 地基** | tools.schema.json + 三端生成器 + 前端 parseToolCalls 重構 + 評測腳本 per-task 化 | 協議 v2 spec（不改模型） | 2-3 天 |
| **P1 日期+轉帳+欠款** | T6 date 強化 → T1+T2+T3 資料生成 → Stage A SFT → 守門+per-task 評測 → 量化上傳 → 主專案 5.1-5.3 | v2.0 模型 + 三工具可用 | 約 1.5 週 |
| **P2 週期+多筆** | T4+T5 → Stage B → 多筆預覽 UI 定稿 | v2.1 模型 | 約 1 週 |
| **P3 查詢規劃器** | T7-T9 → Stage C → 報表頁 AI 入口 | v2.2 模型 + 口語報表 | 約 1 週 |
| **P4 進階（可選）** | T10-T13 擇一 | 研究深度 + 差異化 | 視結果 |

### 守門指標（每階段必過，否則回退資料配比重訓）
| 指標 | 現況 | 下限 |
|------|------|------|
| add_record 格式通過 | 98.68% | ≥ 97% |
| add_record EM | 39.35% | ≥ 36%（-3pp 容差） |
| add_record 金額 | 92.30% | ≥ 90% |
| 新 tool 格式通過 | — | ≥ 95% |
| 新 tool 工具選擇正確率（不誤呼 add_record） | — | ≥ 90% |

---

## 7. 風險與緩解

| 風險 | 影響 | 緩解 |
|------|------|------|
| 58M 容量瓶頸：每加一 tool 稀釋既有能力 | add_record 退化 | 40-50% 舊資料混合 + 守門指標 + 分階段只加 1-2 個 |
| 多筆批次（T5）格式崩壞 | 1.3% 格式錯誤 × N 筆 = 批量錯誤 | 預覽卡確認制；T5 放在 P2 而非 P1 |
| 聯絡人 enum 過長 | 模型選錯/編造 | 截斷 top-12 + validator 模糊匹配 + needsConfirmation |
| 日期是既有弱點（61%） | 新 tool 多數依賴日期 | T6 前置，先修再擴 |
| 學術/產品分岔 | 兩套協議維護成本 | 產品線鎖定 JSON；compressed 只留研究 track |
| q4_0 量化後新 tool 表現掉太多 | 端側體驗差 | 每階段都測 q4_0（不只 fp16），評測腳本加 `--quant` 維度 |

---

## 8. 研究價值（順手強化論文）

現況故事：「protocol compression 降低 tool-calling 冗餘」——但只有 1 個 tool，"tool-calling" 名不副實。擴充後可加：
1. **多工具選擇實驗**：工具數 N（1→2→4→6）× 格式（JSON/Compressed）的選擇正確率與格式穩定性矩陣
2. **按情境注入 vs 全量注入** 的消融（prompt 長度 × 58M 的失焦曲線）
3. **per-tool protocol compression**：不同 tool 的槽位結構差異（固定槽 vs 動態 enum vs 開放文字）對壓縮效益的影響
4. 多筆批次 = 一個 prompt 多次 tool call 的穩定性研究（小模型的多動作生成）

這四項全是 58M 端側模型的真實限制研究，比「再加一個 SFT task」有學術辨識度得多。

---

## 9. 立即下一步（建議順序）

1. 寫 `tools.schema.json` + 三端生成器（P0，純程式，不改模型）
2. 日期強化資料生成 + 重測 date 正確率（T6，純資料）
3. 兩件事綠燈後，再開 transfer + debt 的資料生成（P1 主菜）
