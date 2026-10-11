# 專案目錄結構與作用

> 2026-10-11 更新：對齊 v4c 現況。研究敘事與評測結果見 [README.md](README.md)，
> 各世代戰役筆記見 `docs/`（t6 日期強化、t7 短語強化）。

## 核心文件
* `README.md` - 專案門面：模型規格、研究假設、消融實驗、v1~v4c 評測結果與運行指南。
* `report.md` - 30M 時代消融研究完整報告（歷史文件，方法論參考）。
* `academic_report_plan.md` - 學術推甄作品集規劃藍圖（歷史文件）。
* `requirements.txt` - Python 依賴清單。
* `.env.example` - 環境變數與 LLM API 配置範例檔（v1 資料生成用）。

## 資料產生器（gen/，現行產線）
* `gen/build_prompt.py` - system prompt 組裝器：tools schema 注入 + 「今天是 YYYY-MM-DD（星期X）」日期錨定，與 PWA 前端 `aiService.generateSystemPrompt` 逐字對齊。
* `gen/build_v3_dataset.py` - v3 日期強化：星期注入（一律由日期重算）+ 純日偏移白名單增廣，零隨機性可重現。
* `gen/build_short_phrase.py` - v4 短語型樣本（物品+金額，如「早餐60」），train/test 商品詞零重疊設計。
* `gen/build_v4c.py` - v4c 短語強化：商品詞全量入池、破先驗極端金額、小數萬/複合千書寫、全新詞泛化探針（probe_unseen）。
* `gen/mix_v2_dataset.py` - v2 多工具（transfer/debt）資料混合。
* `gen/gen_js_tools.py` - tools.schema.json → 主專案 `toolsSchema.js` 同步產生器。
* `gen/test_build_prompt.py` - 星期映射金標回歸鎖（防 off-by-one 復發）。

## 資料集（dataset/，純 git 追蹤，無 LFS）
* `train_strata.jsonl` / `test_strata.jsonl` - v1 訓練/測試集（3,323/831；`.bak` 為原版守門集）。
* `qwen_full/`、`qwen_batch2/` 等 - v1 時代 Qwen 生成原始/過濾語料。
* `v2/` - v2 多工具資料。
* `v3/` - v3 日期強化（train 4279 / test 493 / gate_v3_weekday 831）。
* `v4/` - v4 短語混合（train_v4 5502 / val 673 / short_test 103 / gate_sub300）。
* `v4c/` - v4c 最終版（train_v4c 7902 / val_v4c 856 / regression_v4 103 / probe_unseen 160）。

## 模型與分詞器
* `model/model.py` - 58M Transformer 架構（GQA, RoPE, SwiGLU, Tied Embedding）。
* `model/tokenizer.py` - 自訂分詞器載入與 8 個記帳特殊 Token 註冊（vocab 6406）。

## 訓練與驗證
* `restore_hf_checkpoint.py` - HF `the-walking-fish/jijun-LM` safetensors → 本機 `.pt` 權重返還（continue-SFT 起點；Llama 命名 → 本機命名映射，111 tensors）。
* `train_custom_sft.py` - SFT 微調（全量/continue）；內建 bf16 AMP（`--amp`）與進程級 VRAM 硬上限（`--vram_fraction`，保桌面餘裕）。
* `train_custom_pretrain.py` - 無監督 Wikipedia + 對話混合預訓練（Packing + 比例控制）。
* `dataset.py` - PyTorch Dataset：ChatML 拼接、Loss Masking、日期前綴處理（見 docs/t7 之 date bug 教訓）。
* `eval_checkpoints.py` - 逐欄位 greedy 評測：任意 `.pt` × 任意 jsonl → format/EM/六欄位正確率 + 失敗範例。
* `evaluate_benchmark.py` - v1 分層量化評測（Reward/延遲）。
* `verify_model.py` - 15 句手寫中文查詢語意抽查。
* `trainer/validate_json.py` - 與 PWA 前端對齊的解析器與 RL 獎勵函數。
* `demo.py` - Gradio 互動記帳測試網頁。
* `export_model.py` - ONNX 導出 + Llama GGUF 權重重映射。
* `export_all_quants.py` - 多精度量化導出（GGUF 上傳 HF 用）。

## 訓練資料管線（v1 時代，保留供重現）
* `generate_dataset.py` - vllm Qwen3.6-27B API 批量口語樣本生成（斷點續傳、`--bias_categories`）。
* `filter_dataset.py` - 舉證責任倒置語意過濾。
* `split_dataset.py` - 80/20 切分。
* `convert_dataset_dates.py` - 日期標準化。

## 模型權重（saves/，.gitignore 不入庫）
* 本機 checkpoint：`saves/`、`saves_v4*/`（`*.pt` 全部 gitignore）。
* 正式發佈管道：Hugging Face `the-walking-fish/jijun-LM`（safetensors）與 `jijun-LM-GGUF`（量化版，PWA wllama 載入）。
* 本地若無 `.pt`，用 `restore_hf_checkpoint.py` 從 HF 返還。

## 文件（docs/）
* `ai-expansion-plan.md` - 工具擴張路線圖（T1~T13，含進度標記）。
* `p0-protocol-v2-spec.md` - Protocol v2（多工具 tool_call）規格。
* `p1-windows-sft-runbook.md` - v2 時代 Windows 訓練 runbook（歷史歸檔；WSL 現行流程見 README 8.4）。
* `t6-v3-dataset-notes.md` - v3 日期強化戰役筆記（星期 off-by-one 修復）。
* `t7-v4c-short-phrase-notes.md` - v4/v4b/v4c 短語強化戰役筆記（date bug、湊整先驗、泛化探針、VRAM 治理）。
