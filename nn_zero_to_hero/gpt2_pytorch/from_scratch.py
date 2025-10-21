import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 384
dropout_p = 0.2
block_size = 256 # context length

class FFN(nn.module):
    def __init__(self):
        super().__init__(n_embd)
        self.net = nn.Sequential( # note: no batch norm here, unlike Vaswani Transformers
            nn.Linear(n_embd, 4 * n_embd), # follow GPT2 paper
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_p),
        )
    def forward(self, x):
        return self.net(x)

class SelfAttentionHead(nn.module):
    def __init__(self, head_size):
        super().__init__()
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.queries(x)
        K = self.keys(x)
        qk_t = Q @ K.transpose(-2,-1) / C**-0.5 # don't forget negative
        wei = qk_t.masked_fill(self.tril[:T, :T] == 0, "-inf") # decoder block, cannot tend to future tokens
        wei = F.softmax(wei, dim=-1) # T,T
        wei = self.dropout(wei)
        
        V = self.values(x)
        out = wei @ V
        return out

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
