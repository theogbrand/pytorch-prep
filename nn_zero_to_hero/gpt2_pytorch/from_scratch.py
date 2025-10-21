import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 384
dropout_p = 0.2

class FFN(nn.module):
    def __init__(self):
        super().__init__(n_embd)
        self.net = nn.Sequential( # note: no batch norm here, unlike Vaswani Transformers
            nn.Linear(n_embd, 4 * n_embd), # follow GPT2 paper
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_p),
        )
    def _forward(self, x):
        return self.net(x)


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
