import os
import json
import argparse
import torch
import torch.nn as nn
from model.model import MiniMindLM, ModelConfig
from model.tokenizer import BookkeepingTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 記帳模型導出工具 (ONNX & GGUF 權重映射)")
    parser.add_argument("--ckpt_path", type=str, default="./saves/best_bookkeeping_model.pt", help="微調後的 PyTorch .pt 權重路徑")
    parser.add_argument("--base_model", type=str, default="jingyaogong/minimind-3", help="基礎 model/tokenizer 路徑")
    parser.add_argument("--out_dir", type=str, default="./exports", help="導出產物儲存目錄")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="是否導出為 ONNX 格式")
    parser.add_argument("--export_gguf_prep", action="store_true", default=True, help="是否進行 GGUF 權重映射重命名")
    return parser.parse_args()

def export_to_onnx(model, tokenizer, config, out_path):
    """
    將 PyTorch 模型導出為 ONNX 格式，並指定動態長度軸以相容 onnxruntime-web
    """
    print(f"正在導出 ONNX 模型至: {out_path} ...")
    model.eval()
    
    # 建立 dummy 輸入 (batch_size=1, seq_len=128)
    dummy_input = torch.randint(0, config.vocab_size, (1, 128), dtype=torch.long)
    
    # 指定動態軸，以利於在瀏覽器端推論時接受不同長度的 Prompt
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            out_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,  # 使用 opset 17/18 以獲得較佳的 Transformer 算子支援
            do_constant_folding=True
        )
    print("ONNX 導出成功！")

    # 4-bit / 8-bit 量化
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = out_path.replace(".onnx", "_q4.onnx")
        print(f"正在對 ONNX 模型進行 8-bit 動態量化，輸出路徑: {quantized_path} ...")
        quantize_dynamic(
            model_input=out_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8
        )
        print("ONNX 量化模型導出成功！體積已壓縮。")
    except ImportError:
        print("提示: 未安裝 onnxruntime.quantization，跳過 ONNX 4-bit 量化步驟。")

def convert_to_llama_state_dict(custom_state_dict, config):
    """
    學術與工程核心貢獻：
    將我們自建的 MiniMind 自定義結構權重，重映射並命名為標準 Llama/Qwen 的權重名稱。
    這能讓 llama.cpp 的 convert_hf_to_gguf.py 與 transformers 官方庫直接辨識我們訓練出來的模型！
    """
    llama_state_dict = {}
    
    # 定義映射規則對照表
    # 格式: custom_prefix -> llama_prefix
    mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight"
    }

    # 1. 處理基礎映射
    for k, v in custom_state_dict.items():
        if k in mapping:
            llama_state_dict[mapping[k]] = v

    # 2. 處理逐層的 Transformer 區塊映射
    # MiniMind Layer 結構對照 Llama Decoder Layer
    for i in range(config.n_layers):
        custom_layer_prefix = f"layers.{i}."
        llama_layer_prefix = f"model.layers.{i}."

        # RMSNorm 映射
        llama_state_dict[f"{llama_layer_prefix}input_layernorm.weight"] = custom_state_dict[f"{custom_layer_prefix}attention_norm.weight"]
        llama_state_dict[f"{llama_layer_prefix}post_attention_layernorm.weight"] = custom_state_dict[f"{custom_layer_prefix}ffn_norm.weight"]

        # Attention 機制權重映射
        llama_state_dict[f"{llama_layer_prefix}self_attn.q_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}attention.wq.weight"]
        llama_state_dict[f"{llama_layer_prefix}self_attn.k_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}attention.wk.weight"]
        llama_state_dict[f"{llama_layer_prefix}self_attn.v_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}attention.wv.weight"]
        llama_state_dict[f"{llama_layer_prefix}self_attn.o_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}attention.wo.weight"]

        # SwiGLU FFN 權重映射
        llama_state_dict[f"{llama_layer_prefix}mlp.gate_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}feed_forward.w1.weight"]
        llama_state_dict[f"{llama_layer_prefix}mlp.down_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}feed_forward.w2.weight"]
        llama_state_dict[f"{llama_layer_prefix}mlp.up_proj.weight"] = custom_state_dict[f"{custom_layer_prefix}feed_forward.w3.weight"]

    return llama_state_dict

def prepare_gguf_export(ckpt_path, base_model, out_dir, config):
    """
    載入我們的微調權重，重映射並保存為符合 Hugging Face LlamaForCausalLM 結構的權重目錄。
    這樣 llama.cpp 就能無痛讀取此目錄，將其轉換為 GGUF 格式。
    """
    print("正在準備 GGUF 轉換所需的 Llama 權重對齊...")
    
    # 載入我們的微調權重
    try:
        custom_weights = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"錯誤: 無法載入權重檔案 {ckpt_path}: {e}")
        return

    # 重映射為 Llama 格式
    llama_weights = convert_to_llama_state_dict(custom_weights, config)
    
    # 建立輸出目錄
    hf_dir = os.path.join(out_dir, "llama_compatible_weights")
    os.makedirs(hf_dir, exist_ok=True)
    
    # 保存重映射後的權重
    weight_save_path = os.path.join(hf_dir, "pytorch_model.bin")
    torch.save(llama_weights, weight_save_path)
    print(f"Llama 對齊權重已儲存至: {weight_save_path}")
    
    # 寫入 Llama / Qwen 相容的 config.json
    # 這能讓 llama.cpp 的 convert_hf_to_gguf.py 知道模型的架構尺寸
    llama_config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": config.d_model,
        "initializer_range": 0.02,
        "intermediate_size": int(2 * (config.d_model * 8 // 3) / 3), # SwiGLU 中間層大小
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.n_heads,
        "num_hidden_layers": config.n_layers,
        "num_key_value_heads": config.n_kv_heads,
        "rms_norm_eps": config.norm_eps,
        "rope_scaling": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "vocab_size": config.vocab_size
    }
    
    config_save_path = os.path.join(hf_dir, "config.json")
    with open(config_save_path, "w") as f:
        json_str = json.dumps(llama_config, indent=2)
        f.write(json_str)
    print(f"HuggingFace config.json 已寫入: {config_save_path}")
    
    print("\n=== Llama-GGUF 權重映射成功 ===")
    print("您現在可以在終端機執行 llama.cpp 的腳本，將此目錄直接轉成 GGUF 格式：")
    print(f"python llama.cpp/convert_hf_to_gguf.py {hf_dir} --outtype q4_0 --outfile {out_dir}/minimind_bookkeeping.gguf")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化 Tokenizer 與配置
    tokenizer = BookkeepingTokenizer(args.base_model)
    config = ModelConfig(vocab_size=tokenizer.vocab_size)

    # 載入主模型
    model = MiniMindLM(config)
    
    # 嘗試載入微調權重
    if os.path.exists(args.ckpt_path):
        try:
            model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
            print(f"已成功載入微調權重: {args.ckpt_path}")
        except Exception as e:
            print(f"警告: 載入權重失敗 ({e})，將以隨機權重進行結構導出測試。")
    else:
        print(f"警告: 未找到微調權重 {args.ckpt_path}，將以隨機權重進行結構導出測試。")

    # 1. 執行 ONNX 導出
    if args.export_onnx:
        onnx_out_path = os.path.join(args.out_dir, "minimind_bookkeeping.onnx")
        export_to_onnx(model, tokenizer, config, onnx_out_path)

    # 2. 執行 GGUF 權重映射對齊
    if args.export_gguf_prep:
        prepare_gguf_export(args.ckpt_path, args.base_model, args.out_dir, config)

if __name__ == "__main__":
    main()
