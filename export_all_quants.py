import os
import json
import torch
import numpy as np
import gguf

# 欲產出的量化規格定義
TARGET_QUANTS = [
    ("fp16", gguf.GGMLQuantizationType.F16, gguf.LlamaFileType.MOSTLY_F16, "bookkeeping_model_f16.gguf"),
    ("q8_0", gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0, "bookkeeping_model_q8_0.gguf"),
    ("q6_k", gguf.GGMLQuantizationType.Q6_K, gguf.LlamaFileType.MOSTLY_Q6_K, "bookkeeping_model_q6_k.gguf"),
    ("q5_0", gguf.GGMLQuantizationType.Q5_0, gguf.LlamaFileType.MOSTLY_Q5_0, "bookkeeping_model_q5_0.gguf"),
    ("q4_0", gguf.GGMLQuantizationType.Q4_0, gguf.LlamaFileType.MOSTLY_Q4_0, "bookkeeping_model_q4_0.gguf"),
]

def export_single_quant(weights_dir, out_dir, name, qtype, file_type, filename):
    config_path = os.path.join(weights_dir, "config.json")
    bin_path = os.path.join(weights_dir, "pytorch_model.bin")
    out_gguf_path = os.path.join(out_dir, filename)

    if not os.path.exists(bin_path) or not os.path.exists(config_path):
        print(f"Error: Missing weights {bin_path} or config {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    state_dict = torch.load(bin_path, map_location="cpu")
    writer = gguf.GGUFWriter(out_gguf_path, "llama")

    writer.add_block_count(config["num_hidden_layers"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_file_type(file_type)

    for tensor_name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)

        # 僅對 2D 以上的權重矩陣且欄數為 32 倍數者進行 Block 量化；其餘 (如 1D norm/bias) 維持 F16
        if qtype == gguf.GGMLQuantizationType.F16:
            quantized_arr = arr.astype(np.float16)
        elif arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and "norm" not in tensor_name:
            try:
                quantized_arr = gguf.quantize(arr, qtype)
            except Exception as e:
                print(f"Warning: Failed to quantize {tensor_name} with {name} ({e}), falling back to F16")
                quantized_arr = arr.astype(np.float16)
        else:
            quantized_arr = arr.astype(np.float16)

        writer.add_tensor(tensor_name, quantized_arr, raw_dtype=qtype if (arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and "norm" not in tensor_name and qtype != gguf.GGMLQuantizationType.F16) else gguf.GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(out_gguf_path) / (1024 * 1024)
    print(f"[{name.upper()}] Exported successfully -> {out_gguf_path} ({size_mb:.2f} MB)")

def main():
    weights_dir = "./jijun-LM-GGUF/llama_compatible_weights"
    out_dir = "./jijun-LM-GGUF"
    os.makedirs(out_dir, exist_ok=True)

    print("=== Starting GGUF Multi-Quantization Export (q4_0, q5_0, q6_k, q8_0, fp16) ===")
    for name, qtype, file_type, filename in TARGET_QUANTS:
        export_single_quant(weights_dir, out_dir, name, qtype, file_type, filename)

    print("\nAll 5 GGUF quantized models successfully generated!")

if __name__ == "__main__":
    main()
