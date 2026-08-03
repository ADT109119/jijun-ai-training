import os
import json
import torch
import numpy as np
import gguf
from safetensors.torch import load_file
from transformers import AutoTokenizer

TARGET_QUANTS = [
    ("fp16", gguf.GGMLQuantizationType.F16, gguf.LlamaFileType.MOSTLY_F16, "bookkeeping_model_f16.gguf"),
    ("q8_0", gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0, "bookkeeping_model_q8_0.gguf"),
    ("q6_k", gguf.GGMLQuantizationType.Q6_K, gguf.LlamaFileType.MOSTLY_Q6_K, "bookkeeping_model_q6_k.gguf"),
    ("q5_0", gguf.GGMLQuantizationType.Q5_0, gguf.LlamaFileType.MOSTLY_Q5_0, "bookkeeping_model_q5_0.gguf"),
    ("q4_0", gguf.GGMLQuantizationType.Q4_0, gguf.LlamaFileType.MOSTLY_Q4_0, "bookkeeping_model_q4_0.gguf"),
]

TENSOR_NAME_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

def translate_tensor_name(name):
    if name in TENSOR_NAME_MAP:
        return TENSOR_NAME_MAP[name]
    
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer_idx = parts[2]
        sub = ".".join(parts[3:])
        
        mapping = {
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }
        if sub in mapping:
            return f"blk.{layer_idx}.{mapping[sub]}"
    
    return name

def export_gguf_from_hf(model_dir, out_dir, name, qtype, file_type, filename):
    config_path = os.path.join(model_dir, "config.json")
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    out_gguf_path = os.path.join(out_dir, filename)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 載入分詞器與詞表
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    tokens = []
    scores = []
    toktypes = []

    for token_text, token_id in sorted_vocab:
        tokens.append(token_text)
        scores.append(0.0)
        if token_text in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            toktypes.append(3) # CONTROL
        else:
            toktypes.append(1) # NORMAL

    state_dict = load_file(safetensors_path)
    writer = gguf.GGUFWriter(out_gguf_path, "llama")

    # 1. 寫入 Llama 架構屬性
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_file_type(file_type)

    # 2. 寫入完整 BPE Tokenizer 詞表元資料 (gpt2 模式以適應 Minimind/Llama-BPE 詞表)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_bos_token_id(config.get("bos_token_id", 1))
    writer.add_eos_token_id(config.get("eos_token_id", 2))

    tok_json_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r", encoding="utf-8") as f:
            tok_json = json.load(f)
        merges_raw = tok_json.get("model", {}).get("merges", [])
        merges = []
        for m in merges_raw:
            if isinstance(m, list):
                merges.append(" ".join(m))
            elif isinstance(m, str):
                merges.append(m)
        if merges:
            writer.add_token_merges(merges)

    # 3. 轉譯並寫入權重 Tensors (轉換 HF 名稱為 GGML 名稱)
    for hf_name, tensor in state_dict.items():
        ggml_name = translate_tensor_name(hf_name)
        arr = tensor.detach().cpu().numpy().astype(np.float32)

        if qtype == gguf.GGMLQuantizationType.F16:
            if "norm" in ggml_name:
                quantized_arr = arr.astype(np.float32)
                raw_type = gguf.GGMLQuantizationType.F32
            else:
                quantized_arr = arr.astype(np.float16)
                raw_type = gguf.GGMLQuantizationType.F16
        elif arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and "norm" not in ggml_name:
            try:
                quantized_arr = gguf.quantize(arr, qtype)
                raw_type = qtype
            except Exception:
                quantized_arr = arr.astype(np.float32)
                raw_type = gguf.GGMLQuantizationType.F32
        else:
            quantized_arr = arr.astype(np.float32)
            raw_type = gguf.GGMLQuantizationType.F32

        writer.add_tensor(ggml_name, quantized_arr, raw_dtype=raw_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(out_gguf_path) / (1024 * 1024)
    print(f"[{name.upper()}] Exported successfully -> {out_gguf_path} ({size_mb:.2f} MB)")

def main():
    model_dir = "./hf_upload/model-compressed-token"
    out_dir = "./jijun-LM-GGUF"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Converting HF Repo ({model_dir}) to GGML-GGUF ===")
    for name, qtype, file_type, filename in TARGET_QUANTS:
        export_gguf_from_hf(model_dir, out_dir, name, qtype, file_type, filename)

    print("\nAll GGUF models converted with GGML tensor naming successfully!")

if __name__ == "__main__":
    main()
