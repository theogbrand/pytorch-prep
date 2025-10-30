import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

n_embd = 384
dropout_p = 0.2
block_size = 256 # context length
n_layers = 6
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
        self.net(nn.Sequential([
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout_p)
        ]
        ))
    def forward(self, x):
        x = self.net(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.queries = nn.Linear(n_embd, head_size)
        self.keys = nn.Linear(n_embd, head_size)
        self.values = nn.Linear(n_embd, head_size)
        self.register_buffer("tril", torch.tril(torch.ones[block_size, block_size]))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.queries(x) 
        K = self.keys(x)
        V = self.values(x)
        qk_t = Q @ K.transpose(1,2) * self.head_size**-0.5 # TODO -> masked_fill comes after this***
        wei = qk_t.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # B,T,T
        wei = F.softmax(wei, dim=-1) # over the Keys
        out = self.dropout(wei) # before matmul with V
        wei = wei @ V
        return out

class MHA(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads)) 
        self.proj = nn.Linear(n_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1) # divy then join back
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MHATransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads whole"
        self.ffn = FFN(n_embd)
        self.mha = MHA(n_heads, head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        out = x + self.ffn(self.ln2(x))
        return out

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd) # TODO input is vocab
        self.postion_embd = nn.Embedding(block_size, n_embd) # input is ctx length
        self.blocks = nn.Sequential(*[MHATransformerBlock(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd) 
        self.lm_head = self.Linear(n_embd, vocab_size)

    def forward(self, ix, targets=None): 
        B, T = ix.shape
        te = self.token_embd(ix)
        pe = self.position_embd(torch.arange(T), device=device) # Check back
        x = te + pe
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None: # TODO
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, ix, max_new_tokens=100):
        for _ in range(max_new_tokens):
            cond_ix = ix[:, -block_size:] # remember
            logits, _ = self(cond_ix) # B, T, C
            proba = F.softmax(logits, dim=-1) # TODO softmax over the vocab
            proba = proba[:,-1,:]
            next_idx = torch.multinomial(proba, num_samples=1)
            ix = torch.cat([ix, next_idx])
        return ix


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # can only attend to max block size
    x = torch.stack([data[idx:idx + block_size] for idx in ix])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in ix])
    x,y = x.to(device), y.to(device)
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
print(decode(m.generate(ctx, max_output_tokens=500)[0].tolist()))