# P0 研究報告：Protocol v2 地基 — 現況審計、規格草案與決策點

> 日期：2026-08-21｜產出：tools.schema.json + gen/build_prompt.py + gen/test_build_prompt.py
> 驗證狀態：400 筆真實訓練資料位元級再現 PASS（見下）

---

## 1. 現況審計：tool 定義散落點（逐處對照）

| # | 位置 | 內容 | 格式特徵 |
|---|------|------|---------|
| 1 | `tools/jijun-ai-training/generate_dataset.py` L466-484 | 訓練資料 tool_definition（每筆樣本動態 sub-sample enum） | spaced JSON（json.dumps 預設）、**無星期**、description = "ISO 8601 格式日期，僅日期精度，例如 YYYY-MM-DD" |
| 2 | `jijun/src/js/aiService.js` L399-416 | 前端 generateSystemPrompt（生產 prompt） | **compact JSON**（JSON.stringify）、**有星期**（`今天是 X（星期Y）`）、date description 缺「僅日期精度」四字 |
| 3 | `tools/jijun-ai-training/trainer/validate_json.py` L4-141 | 解析器 + validator（訓練端與前端「對齊」的權威副本） | 硬編 add_record 欄位假設（required 4 欄、type enum、date regex） |
| 4 | `jijun/src/js/aiService.js` L484-571 | extractToolCall（前端解析器） | 雙軌：compressed [AMT][CAT] + JSON；bare-args fallback 隱含假設 add_record |
| 5 | `tools/jijun-ai-training/demo_gguf.py` L26-40 / `verify_model.py` L14-20 | 推論端 prompt（第三處 TOOL_DEF 手抄） | spaced JSON、有星期（demo）/ 無星期（verify） |

**審計結論：**
- 5 處獨立副本，其中 3 處是「prompt 產生端」（#1/#2/#5），2 處是「解析驗證端」（#3/#4）
- 解析端 #3/#4 目前互相一致（前端有刻意對齊註解），但欄位假設硬編，新 tool 加入時兩端都要人肉改
- **發現 2 個既有位元級不一致**（v1 模型對兩者都已適應，格式通過率 98.68% 是前端 compact+星期 格式下測得，所以兩者皆「可工作」，但 P0 必須定 canonical）：
  1. **星期**：訓練資料 1672 筆全部無星期；前端生產 prompt 有星期
  2. **JSON 空位**：訓練資料 spaced（`"name": "add_record"`）；前端 compact（`"name":"add_record"`）

## 2. Protocol v2 規格草案（待批核）

### 2.1 輸出格式（最大決策點）

- v1：單一 `<tool_call>{"name": "add_record", "args": {...}}</tool_call>`
- v2：**1..N 個獨立 block 序列**，block 之間容許空白/換行，**不用 JSON 陣列包裹**
  ```
  <tool_call>{"name": "add_record", "args": {...}}</tool_call>
  <tool_call>{"name": "add_record", "args": {...}}</tool_call>
  ```
- 理由：58M 生成「獨立完整 JSON 物件」比生成「list 結構（[ ] + 逗號分隔）」穩定；每個 block 自含 name envelope，parser 可用全域 regex `/<tool_call>([\s\S]*?)<\/tool_call>/g` 一次抓完
- v1 模型輸出 = 1 個 block 的 v2，**完全向後相容**
- Compressed 軌（[AMT][CAT]...）：研究用，僅 add_record、僅單 call，不改
- 待確認：bare-args（無 name envelope）在 v2 是否仍接受 → 建議：接受且僅在該情境的 toolset 含 add_record 時解釋為 add_record（保留 v1 容錯），其他 tool 缺 name 一律 invalid

### 2.2 Prompt 格式

- 單 tool（v1）：`今天是 {date}。你是一個記帳助理。你被賦予了以下 tools:\n{單一物件}` — 維持現況
- 多 tool（v2，P1 模型起）：tools 部分改 **JSON 陣列** `[{tool1}, {tool2}, ...]`，其餘前綴不變
- Canonical 序列化 = spaced（與全部現有訓練資料一致）；前端 v1-compat 模式維持 compact（不擾動現役模型輸入位元），v2 模型上線時前端切 spaced（與 P1 訓練資料一致）
- **星期**：建議 v2 批次統一為「有星期」（生產端已有、對相對日期解析有益），T6 日期強化批次用 data 面確認（加/減星期各測一輪 date 正確率）
- 動態 enum：schema 標記 `enum_source`（category/account/contact），生成時注入實際清單；contact top-12 截斷
- `validation` 區塊（gt0、date format、custom rules）只供 validator，**絕不進 prompt**（prompt 位元穩定 + 省 token）

### 2.3 Validator 語義（parseToolCalls 回傳結構）

```js
parseToolCalls(responseText, ctx) → {
  calls: [{
    name, args,            // 解析結果（dynamic enum miss 且模糊匹配成功時，args 已換成池內值 + 附 resolvedFrom）
    status: 'ok' | 'needs_confirmation' | 'invalid',
    issues: ['缺少必要欄位: date', ...]
  }],
  format: 'json' | 'compressed'
}
```
- 靜態檢查（required/型別/static enum/amount>0/date format）失敗 → `invalid`
- 動態 enum 不在池 → 模糊匹配（normalization + 子字串 + 編輯距離）命中 → ok 附 `resolvedFrom`；未命中 → **`needs_confirmation`**（不是 invalid！UI 讓使用者選，這是產品層救回率，不是格式錯誤）
- 未知 tool name（模型幻覺）→ 該 call invalid，issue = "未知 tool"
- **invalid/needs_confirmation 的 call 絕不自動執行**（產品鐵律：多筆一律過預覽卡確認）
- 向後相容：`extractToolCall(text)` 保留，實作改為 `parseToolCalls` 取第一筆 call 的 args（add_record 語義 + 原有 throw 行為），現有 aiService.test.js 零改動

### 2.4 評測 per-task 化

- `extract_tool_call` → 新增 `extract_tool_calls`（複數，回傳 [(name, args), ...]）；舊函數保留
- 比對欄位改為 **schema 驅動**：取 label args 的 keys 逐欄比對（不再硬編 5 欄）；新 tool 的 tool_selection_accuracy = predicted name == label name
- dataset 新增 `"task": "add_record"` 欄位（舊資料缺省 = add_record，**不改舊檔案**，評測端 fallback）
- 守門集 = 現有 test_strata 原文不動（add_record guardrail）；新 tool 各自 test set 獨立
- 評測腳本加 `--quant` 維度（q4_0 vs fp16 都測，對齊規劃書風險表）

## 3. 已完成的 P0 產出（本次）

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `tools.schema.json` | 單一真相源：6 tools（add_record/transfer/add_debt/settle_debt/add_recurring/query_stats）+ 動態 enum 來源 + 情境注入映射 + 守門指標 | ✅ 已寫（local only） |
| `gen/build_prompt.py` | schema → system prompt 生成器（v1 spaced / compact 變體 / weekday / 多工具陣列） | ✅ 已寫 |
| `gen/test_build_prompt.py` | 黃金測試：400 筆真實訓練資料位元級再現 + compact 前端變體 + 多工具結構 | ✅ **PASS** |
| `docs/p0-protocol-v2-spec.md` | 本檔 | ✅ |

**位元級相容證明**（核心風險已排除）：
```
[PASS] 400 筆真實訓練資料全部位元級再現（spaced v1 格式 + compact 前端變體）
[OK] 多工具格式：JSON 陣列、順序保留、無 enum_source/validation 洩漏
```

## 4. 待用戶決策（3 項）

1. **多工具 prompt 用 JSON 陣列**（2.2）— 替代方案：維持單一物件但 description 寫「可呼叫多次」？陣列是業界慣例且 parser 端零歧義，建議採陣列
2. **v2 前綴帶星期**（2.2）— 建議帶；需 T6 批次用 data 面 A/B 確認 date 正確率
3. **bare-args 容錯**（2.1）— 建議保留（v1 容錯），P1 模型訓練資料全部帶 name envelope 使其自然消失

## 5. P0 實施完成狀態（2026-08-21 全部完成並驗證）

| 項目 | 位置 | 狀態 |
|------|------|------|
| `src/js/toolsSchema.js`（schema const + buildSystemPrompt + parseToolCalls + validateCall + fuzzyMatch） | jijun | ✅ 29 個新單元測試全過 |
| aiService.js 接線：extractToolCall 改 wrapper、generateSystemPrompt 改 builder（v1 輸出 bit-identical 已驗證） | jijun | ✅ 既有 20 tests 零回歸 |
| `tests/unit/toolsSchema.test.js`（多筆、compressed、thinking、bare-args、enum miss、未知 tool、conditional） | jijun | ✅ |
| trainer/validate_json.py：extract_tool_calls + validate_tool_call（schema 驅動，與前端語義一致） | training | ✅ 冒煙測試通過 |
| evaluate_benchmark.py per-task 化 + --quant + torch lazy import（mock 模式免 torch） | training | ✅ dummy 冒煙通過 |
| gen/gen_js_tools.py 同步生成器（--check 漂移偵測，CI 可掛） | training | ✅ |

**全量驗證（2026-08-21）：**
- `npm test`：1598/1598（39 files；基線 1569 + 29 新）
- `npm run build`：成功
- `gen/test_build_prompt.py`：400 筆真實訓練資料位元級再現
- `evaluate_benchmark.py --model_path dummy`：per-task 報告正常（add_record + transfer）
- v1 prompt bit-identical：buildSystemPrompt vs 舊內嵌格式逐位元比對通過

**不做（P0 範圍外）**：aiService 的 executor/UI 預覽卡（P1 實作）、max_tokens 384（P2 多筆才需要）、模型任何改動

