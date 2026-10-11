# T6 日期強化 — v3 資料集（2026-10-10）

## 背景

route map（ai-expansion-plan.md）第 9 章步骤 2：「日期強化資料生成 + 重測 date
正確率（T6，純資料）」。日期是既有弱點（61%），本輪以 476b1e4 審核產出
（train_audited 2296 / test_audited 493）為素材組裝 v3，不動模型權重。

## 本輪發現與修復

### 1. `gen/build_prompt._weekday_zh_for` 星期錯位一天（已修復）

舊實作 `_WEEKDAY_ZH[date.weekday() % 7]`：`date.weekday()` 是 Monday=0..Sunday=6，
但清單以「星期日」開頭 → 星期日誤標星期六、星期一誤標星期日（整體錯位一天）。
前端 `aiService.getCurrentDateAnchor` 用 `getDay()`（Sunday=0）是正確的，
Python 等價寫法是 `isoweekday() % 7`。

- **污染範圍確認**：全量掃描 merged_v2（4154 筆帶星期）全部正確——池資料非經此
  函數生成，bug 未滲入既有訓練資料；但未來任何 `weekday=True` 生成路徑都會中毒。
- **回歸鎖**：`gen/test_build_prompt.py::test_weekday_goldens`（6 金標 + 連續 30 天
  掃描，逐日與前端 getDay() 語意交叉驗證）。

### 2. 守門集與洩漏

- 守門集真身是 `dataset/test_strata.jsonl.bak`（831 筆，原文不動）。
  `qwen_full/test_strata.jsonl`（600）與其重疊僅 81，兩者皆為評測資產。
- v3 train 剔除與守門(831)/test 重疊樣本 1 筆。

## v3 組裝規則（gen/build_v3_dataset.py）

1. **星期注入**：對齊前端 runtime prompt（帶（星期X））。星期一律由日期重算，
   絕不信任來源既有星期；注入後做「移除（星期X）→ 逐位元回溯原文」驗證。
2. **日期偏移增廣**：僅限「純日偏移」相對詞白名單（今天/今日/剛剛/昨天/昨晚/
   前天/明天——label−ref 偏移量在任意同步位移 Δ 下恆等）；含週/月/號/月底等
   曆法邊界詞或絕對日期者拒絕增廣。偏移量 day+2 / week−3 各一筆，ref 與 label
   同步位移，`aug` 頂層欄位記錄（dataset.py 只讀 messages，多餘鍵無害，便於 A/B）。
3. **Schema 靜態驗證**：逐筆過 tools.schema.json
   （name=add_record、required ⊆ keys ⊆ properties、enum ∈ system 池、date 格式、
   type ∈ enum、amount > 0）。
4. 零隨機性：全程確定性組裝，可重現。

## 產出（dataset/v3/）

| 檔 | 筆數 | 說明 |
|---|---|---|
| `train_v3.jsonl` | 4279 | 基礎 2295（星期注入）+ 增廣 1984（白名單樣本 ×2 位移） |
| `test_v3.jsonl` | 493 | test_audited 星期注入版 |
| `gate_v3_weekday.jsonl` | 831 | 守門集星期注入變體（label 零改動，供 v3 模型在 runtime 格式下評測） |

驗證結果：schema 剔除 0；增廣 parent-offset 恆等覆核 1984/1984；全線星期錯位 0。

## 訓練注意（Stage A 混合時）

- 守門評測仍必跑**原版** `test_strata.jsonl.bak`（v1 無星期格式，衡量訓練未見格式
  下的韌性）；`gate_v3_weekday.jsonl` 是「同格式對照」，兩者都要綠燈。
- train_v3 中含 aug 樣本佔比 46.4%，混合舊池資料（merged_v2 4154）時注意
  add_record 基礎分佈不被增廣樣本稀釋失衡——A/B 時可用 `aug` 欄位抽離對比。
- 增廣只覆蓋純日偏移詞；週/月類相對詞（上週/上個月/月底）仍依賴原樣本學習，
  本輪刻意不位移（曆法邊界非等變），此為刻意取捨而非遺漏。
