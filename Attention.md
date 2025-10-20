Vaswani 2017:
```python
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, is_causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        B,T,C = x.shape
        qkv = self.qkv(x).view(B,T,3,self.n_heads,self.head_dim).transpose(1,3)  # [B,h,T,d]
        q,k,v = qkv[:,:, :,0], qkv[:,:, :,1], qkv[:,:, :,2]                      # each [B,h,T,d]

        scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)  # [B,h,T,T]

        if self.is_causal:
            if (self.causal_mask is None) or (self.causal_mask.size(-1) != T):
                mask = torch.tril(torch.ones(T,T, device=x.device, dtype=torch.bool))
                self.causal_mask = ~mask[None,None]                               # True where disallowed
            scores = scores.masked_fill(self.causal_mask, float("-inf"))

        if attn_mask is not None:
            # attn_mask: broadcastable to [B,1,T,T] or [B,1,1,T] (e.g., padding mask on keys)
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)                                          # softmax over keys
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                                               # [B,h,T,d]
        out = out.transpose(1,3).contiguous().view(B,T,C)                         # [B,T,C]
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x), approximate="tanh")))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0, is_causal=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, is_causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model*mlp_ratio), dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

# --- example usage ---
# B,T,C = 2, 8, 64
# x = torch.randn(B,T,C)
# # padding mask on keys (True for pads): shape [B,T] -> convert to additive mask [B,1,1,T]
# key_pad = torch.zeros(B,T, dtype=torch.bool)  # no pads
# additive_mask = key_pad[:,None,None,:].masked_fill(key_pad[:,None,None,:], float("-inf")).to(x.device)
# block = TransformerBlock(d_model=64, n_heads=8, mlp_ratio=4, dropout=0.1, is_causal=True)
# y = block(x, attn_mask=additive_mask)  # [B,T,C]
```