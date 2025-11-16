from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int | None = None
    bias: bool = False
    sliding_window_size: int | None = None
    context_length: int = 256

    def __post_init__(self, d_model, n_heads, n_kv_heads) -> None:
        # TODO: add checks if necessary
        assert d_model % n_heads == 0, "d_model and n_heads needs to be awhole number"
        # n_heads=8 // n_kv_heads=2 = 4 groups
        assert n_heads % n_kv_heads == 0
        ...


class Attention(nn.Module):

    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()

        # Config
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads

        # TODO: can you write what is the number of output features with a fused QKV? # MHA
        # torch.cat([h(x) for h in self.heads])
        self.head_dim = self.d_model // self.n_heads
        # self.mha_proj = nn.Linear(cfg.d_model,self.n_heads*head_dim)
        # total number of heads is the sum of all of them, kv shares multiple q heads -> stretch it out to perform attention ops at one go
        # 8 num_heads, 2 kv_heads, total number of attention_heads = 8q+2k+2v
        self.qkv_proj = nn.Linear(cfg.d_model, (cfg.n_heads+ 2* cfg.n_kv_heads)*head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads*head_dim, cfg.d_model)

    def forward(
        self,
        x: Tensor, # B, T, C//groups or num_heads
    ) -> Tensor:
        b, t, d_model = x.shape
        assert d_model == self.cfg.d_model

        # Project QKV -> [b, t, (cfg.n_heads+ 2* cfg.n_kv_heads)*head_dim]
        qkv = self.qkv_proj(x)

        # TODO: Implement MHSA with causal mask (with sliding window) and GQA
        
        Q,K,V = qkv.reshape(b,t,self.cfg.n_heads, self.head_dim), qkv.reshape(b,t,self.cfg.n_kv_heads, self.head_dim), qkv.reshape(b,t,self.cfg.n_kv_heads,self.head_dim)
        
        scores = 
        mask =  > window_size 
        [float("-inf"),float("-inf"),1,1,1] 
        
        # Question: can you write what's the shape of the output of the "forward" function?



if __name__ == "__main__":
    cfg = AttentionConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        bias=False,
    )

    attn = Attention(cfg)

    B, T = 2, 16
    x0 = torch.randn(B, T, cfg.d_model)
    y0 = attn(x0)

    for i in range(4):
        x_step = torch.randn(B, 1, cfg.d_model)
        y_step = attn(x_step)

    print(f"Model ran, {y_step.shape=}")

