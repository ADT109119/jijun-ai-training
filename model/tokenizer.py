import os
from transformers import AutoTokenizer

class BookkeepingTokenizer:
    """
    自訂記帳分詞器封裝類別
    負責載入基礎中文分詞器，動態註冊記帳專屬的特殊標記 (Special Tokens)，
    並提供壓縮編碼與 XML 標記包裝功能。
    """
    
    # 專用協議特殊標記 (用於省 token 格式策略的消融實驗)
    SPECIAL_TOKENS = {
        "additional_special_tokens": [
            "[AMT]",   # 金額 (Amount)
            "[CAT]",   # 分類 (Category)
            "[ACC]",   # 帳戶 (Account)
            "[DESC]",  # 備註 (Description)
            "[TYPE]",  # 收支類型 (Type: expense/income)
            "<tool_call>",
            "</tool_call>"
        ]
    }

    def __init__(self, base_model_name_or_path: str = "jingyaogong/minimind-3"):
        """
        Args:
            base_model_name_or_path: 基礎 MiniMind/Qwen 模型的路徑或名稱
        """
        # 1. 載入基礎 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path, 
                trust_remote_code=True
            )
        except Exception as e:
            # Fallback: 如果網路不通，使用本地 placeholder 並提示
            print(f"警告: 無法從 HuggingFace 載入基礎分詞器 {base_model_name_or_path}: {e}")
            print("請確保網路通暢，或在本地設定正確的路徑。")
            raise e

        # 2. 向分詞器註冊自訂的特殊標記
        # 這會為分詞器新增獨立 ID，防止這些專用符號被切碎為 bytes 字符
        num_added = self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        print(f"成功向分詞器註冊 {num_added} 個記帳專屬特殊標記。")
        print(f"當前詞表總大小 (vocab_size): {len(self.tokenizer)}")

    @property
    def vocab_size(self) -> int:
        """
        獲取擴充後的詞表大小，用於更新 PyTorch ModelConfig 的 vocab_size
        """
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, max_length: int = 512, truncation: bool = True, add_special_tokens: bool = False) -> list:
        """
        將文字編碼為 Token ID 列表
        """
        return self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: list, skip_special_tokens: bool = False) -> str:
        """
        將 Token ID 列表還原為文字
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def format_compressed_record(self, amount: float, category: str, account: str, description: str, record_type: str) -> str:
        """
        將記帳資料包裝成特殊 Token 壓縮協議格式 (消融實驗核心)
        
        例如:
        <tool_call>[AMT]150[CAT]餐飲[ACC]信用卡[DESC]午餐[TYPE]expense</tool_call>
        """
        return (
            f"<tool_call>[AMT]{amount}"
            f"[CAT]{category}"
            f"[ACC]{account}"
            f"[DESC]{description}"
            f"[TYPE]{record_type}</tool_call>"
        )

    def format_json_record(self, amount: float, category: str, account: str, description: str, record_type: str) -> str:
        """
        將記帳資料包裝成標準 JSON 格式 (對照組)
        
        例如:
        <tool_call>{"amount": 150, "category": "餐飲", "account": "信用卡", "description": "午餐", "type": "expense"}</tool_call>
        """
        record_json = {
            "amount": amount,
            "category": category,
            "account": account,
            "description": description,
            "type": record_type
        }
        import json
        return f"<tool_call>{json.dumps(record_json, ensure_ascii=False)}</tool_call>"

if __name__ == "__main__":
    # 測試擴充效果
    # 註: 需要安裝 transformers 與 sentencepiece
    try:
        tok = BookkeepingTokenizer("jingyaogong/minimind3-sft")
        
        # 測試兩種編碼格式的 Token 數量對比
        test_data = {
            "amount": 150,
            "category": "餐飲",
            "account": "信用卡",
            "description": "麥當勞午餐",
            "record_type": "expense"
        }
        
        json_str = tok.format_json_record(**test_data)
        comp_str = tok.format_compressed_record(**test_data)
        
        json_tokens = tok.encode(json_str)
        comp_tokens = tok.encode(comp_str)
        
        print("\n=== Tokenizer 消融測試 ===")
        print("標準 JSON 格式:", json_str)
        print("JSON Token 數量:", len(json_tokens))
        print("---------------------------------")
        print("特殊 Token 壓縮格式:", comp_str)
        print("壓縮後 Token 數量:", len(comp_tokens))
        print(f"節省比例: {((len(json_tokens) - len(comp_tokens)) / len(json_tokens) * 100):.2f}%")
        
    except Exception as e:
        print("Tokenizer 載入失敗 (通常是本地尚無環境):", e)
