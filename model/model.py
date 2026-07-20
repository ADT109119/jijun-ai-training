import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    """
    模型超參數配置類別
    """
    vocab_size: int = 6400      # 詞彙表大小 (將依自訂 Tokenizer 擴充)
    d_model: int = 512          # 隱藏層維度 (d_model)
    n_layers: int = 8           # Transformer 堆疊層數
    n_heads: int = 8            # Query 注意力頭數
    n_kv_heads: int = 4         # Key/Value 注意力頭數 (GQA: 頭數比為 2:1)
    multiple_of: int = 32       # SwiGLU 中間維度對齊基數
    norm_eps: float = 1e-5      # RMSNorm 的穩定常數 epsilon
    max_seq_len: int = 512      # 最大序列長度
    hidden_dim: Optional[int] = 1536 # SwiGLU 中間層維度 (預設為 1536 以對齊學術規劃，為 None 時公式自適應)
    dropout: float = 0.1     # Dropout 比率 (防止過擬合)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相較於傳統 LayerNorm，它移除了均值對齊 (Mean Center)，只進行方差縮放，能減少 7%~10% 的運算開銷並保持梯度穩定。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可學習的縮放參數 gamma (維度為 d_model)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形狀: [batch_size, seq_len, d_model]
        # 1. 計算每個 token 向量的均方根值 (對最後一個維度計算)
        # variance = x^2 的均值
        # rsqrt = 1 / sqrt(variance + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        # 2. 進行歸一化並乘上可學習的權重 weight (gamma)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_rope_freqs(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    預先計算旋轉位置編碼 (RoPE - Rotary Position Embedding) 的正弦與餘弦頻率矩陣。
    RoPE 透過將 2D 向量進行旋轉，把絕對位置編碼轉化為相對位置的內積關係。
    
    Args:
        dim: 每個注意力頭的維度 (head_dim)
        end: 最大序列長度 (max_seq_len)
        theta: 底數基底 (預設 10000.0)
    """
    # RoPE 僅對頭維度的前半與後半部分配對旋轉，因此計算維度為 dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # [max_seq_len]
    # 外積計算頻率矩陣: [max_seq_len] * [dim // 2] -> [max_seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    
    # 預先算出 polar 形式的實部 (cos) 與虛部 (sin)，用於複數相乘
    # cos, sin 形狀為 [max_seq_len, head_dim // 2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    將預先計算好的 RoPE 旋轉矩陣應用至 Query 與 Key 張量上。
    xq, xk 的輸入形狀: [batch_size, seq_len, n_heads, head_dim]
    """
    # 1. 重塑 xq/xk，將最後的 head_dim 拆成 [head_dim // 2, 2] 以便進行 2D 複數旋轉
    # xq_complex 形狀: [batch_size, seq_len, n_heads, head_dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 2. 依據 2D 旋轉矩陣公式: [x, y] 旋轉 theta 得到 [x*cos - y*sin, x*sin + y*cos]
    # 我們將 cos/sin 的維度廣播 (broadcast) 至與 xq_/xk_ 相容的形狀
    # cos/sin 原形狀: [max_seq_len, head_dim // 2] -> 廣播為 [1, seq_len, 1, head_dim // 2]
    cos = cos[:xq.shape[1]].unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim // 2]
    sin = sin[:xq.shape[1]].unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim // 2]
    
    xq_out = torch.zeros_like(xq_)
    xq_out[..., 0] = xq_[..., 0] * cos - xq_[..., 1] * sin
    xq_out[..., 1] = xq_[..., 0] * sin + xq_[..., 1] * cos
    
    xk_out = torch.zeros_like(xk_)
    xk_out[..., 0] = xk_[..., 0] * cos - xk_[..., 1] * sin
    xk_out[..., 1] = xk_[..., 0] * sin + xk_[..., 1] * cos
    
    # 3. 旋轉完畢後，將形狀還原為 [batch_size, seq_len, n_heads, head_dim]
    return xq_out.flatten(3), xk_out.flatten(3)

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) 機制
    介於 Multi-Head Attention (MHA) 與 Multi-Query Attention (MQA) 之間。
    多個 Query 頭共用一組 KV 頭（例如本配置為 2 個 Q 頭共用 1 個 KV 頭）。
    這能在大幅減少 KV Cache 記憶體佔用（手機端推論效能瓶頸）的同時，保有近乎 MHA 的模型精度。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads  # 每個 KV 頭對應的 Q 頭數量 (例如 2)
        self.head_dim = config.d_model // self.n_heads
        
        # 線性映射層
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. 將輸入向量映射至 Q, K, V 空間
        # xq: [batch_size, seq_len, n_heads * head_dim]
        # xk, xv: [batch_size, seq_len, n_kv_heads * head_dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. 重塑維度，分離出不同的注意力頭，便於後續運算
        # 形狀變更: [batch_size, seq_len, n_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 3. 套用旋轉位置編碼 RoPE
        xq, xk = apply_rope(xq, xk, cos, sin)
        
        # 4. GQA 的核心操作：將 KV 頭進行擴展 (Repeat) 以對齊 Q 頭的個數
        # 如果每個 KV 頭對應 2 個 Q 頭，我們需要將 KV 頭的個數複製 2 倍
        # 形狀: [batch_size, seq_len, n_kv_heads, head_dim] -> [batch_size, seq_len, n_heads, head_dim]
        if self.num_queries_per_kv > 1:
            xk = xk.repeat_interleave(self.num_queries_per_kv, dim=2)
            xv = xv.repeat_interleave(self.num_queries_per_kv, dim=2)
            
        # 5. 進行轉置以利於矩陣相乘
        # 形狀變更: [batch_size, n_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 6. 計算 Attention Scores: (Q * K^T) / sqrt(head_dim)
        # scores 形狀: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 套用因果遮罩 (Causal Mask)，防止模型在自迴歸生成時預知未來的 token
        if mask is not None:
            scores = scores + mask
            
        # 7. Softmax 歸一化得到注意力權重矩陣
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 8. 注意力加權乘上 Value 矩陣: Scores * V
        # output 形狀: [batch_size, n_heads, seq_len, head_dim]
        output = torch.matmul(scores, xv)
        
        # 9. 還原形狀並將所有注意力頭合併，最後投影輸出
        # 轉置與拼接: [batch_size, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class SwiGLU(nn.Module):
    """
    SwiGLU 前向傳播網路 (Feed-Forward Network)
    採用了 Gated Linear Unit (GLU) 與 Swish (SiLU) 激活函數。
    相較於傳統 ReLU FFN，SwiGLU 提供了雙線性閘控機制，能極大增強小模型的非線性表徵能力。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_dim is not None:
            hidden_dim = config.hidden_dim
        else:
            # 計算 SwiGLU 中間層維度，通常為 8/3 乘上 d_model，並對齊 multiple_of 的倍數
            hidden_dim = int(2 * (config.d_model * 8 // 3) / 3)
            hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # w1 負責閥門閘 (Gate)，w3 負責輸入投影
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w1(x)) 是閘門，乘上 w3(x) 的投影，最後經 w2 輸出
        # 形狀變更: [batch_size, seq_len, d_model] -> [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, d_model]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """
    單層 Transformer 區塊 (包含 RMSNorm -> GQA -> 殘差連接 -> RMSNorm -> SwiGLU -> 殘差連接)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-Norm 架構：輸入先經過 Norm 處理，再進入 Attention/FFN，最後加上殘差殘差連接，能有效預防深層模型梯度消失
        h = x + self.dropout(self.attention(self.attention_norm(x), cos, sin, mask))
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out

class MiniMindLM(nn.Module):
    """
    主模型類別: MiniMindLM
    實現了包含 Tied Embedding 詞表與層疊 Transformer 塊的完整自迴歸生成網路。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Word Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # 堆疊 N 層 Transformer Blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # 最終層歸一化
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # 輸出預測頭 (LM Head)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tied Embedding (權重共享)：將輸入 Embedding 層與輸出投影層的權重綁定，
        # 在 <100M 的微型模型中，這能節省 30%~50% 的參數量，並顯著加快詞向量收斂。
        self.tok_embeddings.weight = self.output.weight
        
        # 預先計算旋轉頻率矩陣，避免前向傳播重複運算
        cos, sin = precompute_rope_freqs(
            dim=config.d_model // config.n_heads,
            end=config.max_seq_len
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # tokens 形狀: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape
        
        # 1. 詞表映射至向量空間
        h = self.tok_embeddings(tokens) # [batch_size, seq_len, d_model]
        
        # 2. 獲取對應序列長度的 RoPE 頻率張量
        # 若生成時 seq_len 超出預先計算範圍，動態延伸頻率
        if seq_len > self.cos.size(0):
            new_cos, new_sin = precompute_rope_freqs(
                dim=self.config.d_model // self.config.n_heads,
                end=seq_len,
                theta=10000.0
            )
            cos = new_cos.to(tokens.device)
            sin = new_sin.to(tokens.device)
        else:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
        
        # 3. 建立因果遮罩 (Causal Mask)
        # 上三角矩陣填充負無窮，使得 Softmax 後未來位置權重為 0
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            
        # 4. 逐層通過 Transformer 區塊
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
            
        # 5. 最終歸一化與預測
        h = self.norm(h)
        logits = self.output(h) # [batch_size, seq_len, vocab_size]
        
        # 6. 如果傳入目標標籤 (targets)，計算交叉熵損失 (Loss)
        loss = None
        if targets is not None:
            # logits: [batch_size * seq_len, vocab_size]
            # targets: [batch_size * seq_len]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100, label_smoothing=label_smoothing)
            
        return logits, loss

def count_parameters(model: nn.Module) -> int:
    """
    計算模型的總可訓練參數量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
