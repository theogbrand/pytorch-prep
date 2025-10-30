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
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout_p)
        )
    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.queries = nn.Linear(n_embd, head_size) # CONCAT later which is why out_dim=head_size 
        self.keys = nn.Linear(n_embd, head_size)
        self.values = nn.Linear(n_embd, head_size)
        self.register_buffer("tril", torch.tensor(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        B,T,C = x.shape
        Q,K,V = self.queries(x), self.keys(x), self.values(x)
        scores = Q @ K.transpose(-2,-1) / (K.shape[-1])**0.5
        wei = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ V
        return out

        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)]) 
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.head_size = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd n_heads must be divisible by zero"
        self.ffn = FFN(n_embd)
        self.mha = MultiHeadAttention(n_heads, self.head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, ix, target=None):
        B, T = ix.shape
        te = self.token_embd(ix)
        pe = torch.arange(T, device=device)
        x = te + pe
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if not target:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = logits.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

         

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