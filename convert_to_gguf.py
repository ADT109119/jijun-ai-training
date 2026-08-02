import os
import json
import torch
import numpy as np
import gguf

def export_gguf(weights_dir, out_gguf_path, file_type=gguf.LlamaFileType.MOSTLY_F16):
    config_path = os.path.join(weights_dir, "config.json")
    bin_path = os.path.join(weights_dir, "pytorch_model.bin")

    if not os.path.exists(bin_path) or not os.path.exists(config_path):
        print(f"Error: Missing weights {bin_path} or config {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Loading state_dict from {bin_path}...")
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

    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        if file_type == gguf.LlamaFileType.MOSTLY_F16 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr)

    print(f"Writing GGUF file to: {out_gguf_path} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    file_size_mb = os.path.getsize(out_gguf_path) / (1024 * 1024)
    print(f"GGUF conversion successful: {out_gguf_path} ({file_size_mb:.2f} MB)")

def main():
    weights_dir = "./jijun-LM-GGUF/llama_compatible_weights"
    out_dir = "./jijun-LM-GGUF"
    os.makedirs(out_dir, exist_ok=True)

    # 匯出 F16 版本
    f16_path = os.path.join(out_dir, "bookkeeping_model_f16.gguf")
    export_gguf(weights_dir, f16_path, gguf.LlamaFileType.MOSTLY_F16)

    # 匯出 F32 版本
    f32_path = os.path.join(out_dir, "bookkeeping_model_f32.gguf")
    export_gguf(weights_dir, f32_path, gguf.LlamaFileType.ALL_F32)

if __name__ == "__main__":
    main()
