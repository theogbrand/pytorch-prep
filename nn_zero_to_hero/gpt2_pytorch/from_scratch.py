import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

from nn_zero_to_hero.gpt2_pytorch.gpt2 import learning_rate

n_embd = 384
dropout_p = 0.2
block_size = 256 # context length
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_heads = 6
lr = 1e-4
batch_size = 64
eval_iters = 200

with open(Path(__file__).parent / 'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long) # int64
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # note: no batch norm here, unlike Vaswani Transformers
            nn.Linear(n_embd, 4 * n_embd), # follow GPT2 paper
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_p),
        )
    def forward(self, x):
        return self.net(x)

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.queries(x)
        K = self.keys(x)
        qk_t = Q @ K.transpose(-2, -1) * (self.head_size**-0.5)
        wei = qk_t.masked_fill(self.tril[:T ,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        V = self.values(x)
        out = wei @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) 
        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size  = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffn = FFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x)) 
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(vocab_size, n_embd)
        self.pe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd=n_embd, n_heads=n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # each token corresponding to vocab lookup table

        tok_embd = self.te(idx) # B, T, C
        pos_embd = self.pe(torch.arange(T, device=device)) # T, C - no unsqueeze needed since they align right
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_condi = idx[:, -block_size:] # 
            logits, _ = self(idx_condi) # B,T,vocab_size
            logits = logits[:, -1, :] # only last token's scores -> B, vocab_size 
            probs = F.softmax(logits, dim=-1) # across vocab scores
            next_idx = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat([idx, next_idx], dim=1) # B, T+1
        
        return idx

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # can only attend to max block size
    x = torch.stack([data[idx:idx + block_size] for idx in ix])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in ix])
    x,y = x.to_device(device), y.to_device(device)
    return x, y
    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for s in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xb, Yb = get_batch(s)
            logits, loss = model(Xb, Yb)
            losses[k] = loss.item()
        loss = losses.mean()
        out[s] = loss
    model.train()
    return out

model = GPTLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

max_iters = 5000
eval_interval = 500

for i in range(max_iters):
    if i % eval_interval == 0 or i == max_iters - 1:
        loss = estimate_loss() # train + val loss

    Xb, Yb = get_batch("train")

    logits, loss = model(Xb,Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

ctx = torch.zeros([1,1], dtype=torch.long, device=device) # 1 char, 1 batch
print(decode(m.generate(ctx, max_tokens=500)[0].tolist()))