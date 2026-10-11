"""把 HF jijun-LM safetensors 還原成 train_custom_sft.py 可載入的 .pt (custom 命名)。"""
import sys, torch
sys.path.insert(0, '.')
from safetensors.torch import load_file

llama = load_file('hf_ckpt/model.safetensors')

rev = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}
out = {}
for k, v in llama.items():
    if k in rev:
        out[rev[k]] = v.clone()
    elif k.startswith('model.layers.'):
        rest = k[len('model.layers.'):]
        i, sub = rest.split('.', 1)
        ck = f"layers.{i}."
        m = {
            'input_layernorm.weight': 'attention_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
            'self_attn.q_proj.weight': 'attention.wq.weight',
            'self_attn.k_proj.weight': 'attention.wk.weight',
            'self_attn.v_proj.weight': 'attention.wv.weight',
            'self_attn.o_proj.weight': 'attention.wo.weight',
            'mlp.gate_proj.weight': 'feed_forward.w1.weight',
            'mlp.up_proj.weight': 'feed_forward.w3.weight',
            'mlp.down_proj.weight': 'feed_forward.w2.weight',
        }
        assert sub in m, sub
        out[ck + m[sub]] = v.clone()
    else:
        raise SystemExit(f'unmapped key: {k}')

import os
os.makedirs('saves', exist_ok=True)
torch.save(out, 'saves/jijun_lm_restored.pt')
print('saved', len(out), 'tensors; vocab =', out['tok_embeddings.weight'].shape)
