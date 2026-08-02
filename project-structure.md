# 專案目錄結構與作用

## 核心文件
* [README.md](../README.md) - 專案說明文件，包含模型規格、研究背景、實驗結果與運行指南。
* [project-structure.md](../project-structure.md) - 專案結構描述檔案。
* [requirements.txt](../requirements.txt) - Python 依賴清單。
* [.env.example](../.env.example) - 環境變數與 LLM API 配置範例檔。

## 數據與生成腳本
* [generate_dataset.py](../generate_dataset.py) - Qwen 大模型批量分層口語樣本生成腳本，支持斷點續傳與 `--bias_categories` 低頻類別偏向。
* [filter_dataset.py](../filter_dataset.py) - 嚴格語意對齊過濾腳本（舉證責任倒置，僅刪除明確屬於其他分類的樣本）。
* [split_dataset.py](../split_dataset.py) - 80/20 訓練測試切分腳本。
* [convert_dataset_dates.py](../convert_dataset_dates.py) - 資料集日期標準化轉換腳本。
* [dataset.py](../dataset.py) - PyTorch Dataset 類別，負責 ChatML 拼接與 Loss Masking。
* [dataset/](../dataset) - SFT 訓練與測試語料目錄。
  * `train_strata.jsonl` - 訓練集 (3,323 筆)。
  * `test_strata.jsonl` - 測試集 (831 筆)。
  * `qwen_full/` - Qwen 生成原始/過濾後語料與合併檔 (`raw_generated.jsonl`、`raw_filtered_v2.jsonl`、`raw_filtered_merged_v2.jsonl`)。

## 模型與分詞器
* [model/model.py](../model/model.py) - 58M Transformer 架構（GQA, RoPE, Tied Embedding）。
* [model/tokenizer.py](../model/tokenizer.py) - 自訂分詞器載入與特殊 Token 註冊器（含 `[DATE]` 等）。

## 訓練與驗證
* [train_custom_sft.py](../train_custom_sft.py) - 雙軌全量 SFT 微調腳本。
* [train_custom_pretrain.py](../train_custom_pretrain.py) - 無監督 Wikipedia + 對話混合預訓練腳本。
* [trainer/validate_json.py](../trainer/validate_json.py) - JSON/壓縮標記雙軌解析器與 RL 獎勵函數。
* [evaluate_benchmark.py](../evaluate_benchmark.py) - 分層量化評測腳本。
* [verify_model.py](../verify_model.py) - 語意抽查腳本（15 句手寫中文查詢 vs 模型輸出）。
* [demo.py](../demo.py) - 基於 Gradio 的互動記帳測試網頁。
* [export_model.py](../export_model.py) - ONNX 與 GGUF 重映射導出工具。
* [saves/](../saves) - 模型 Checkpoint 目錄。
  * `best_bookkeeping_model.pt` - 最佳 SFT 模型（demo 使用）。
  * `best_pretrain_model.pt` - 預訓練權重。
  * `backups/` - 定期模型備份。
