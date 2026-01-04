# Basics - Start Here

A focused guide covering the fundamental building blocks for LLM research. Master these topics before diving into the comprehensive [REVISION_TOPICS.md](./REVISION_TOPICS.md).

---

## 1. Single Neuron Backpropagation

### Core Concept
Understanding how gradients flow through a single neuron is the foundation of deep learning.

### Forward Pass
```python
# Single neuron computation
z = w * x + b
a = activation(z)  # e.g., sigmoid, ReLU
```

### Backward Pass
```python
# Chain rule application
dL/dw = dL/da * da/dz * dz/dw
dL/db = dL/da * da/dz * dz/db
dL/dx = dL/da * da/dz * dz/dx
```

### Key Concepts
- **Chain rule**: Multiply gradients backward through the computation graph
- **Local gradients**: Each operation computes its local derivative
- **Gradient accumulation**: Sum gradients from multiple paths

### Implementation Practice
- Implement forward and backward pass without autograd
- Verify gradients using numerical differentiation
- Understand gradient flow for common activations (ReLU, sigmoid, tanh)

---

## 2. Positional Encodings

### Why Needed?
Transformers have no inherent notion of token position - attention is permutation-invariant. Positional encodings add position information.

### Types of Positional Encodings

#### 2.1 Sinusoidal (Original Transformer)
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Fixed, not learned
- Can extrapolate to longer sequences
- Used in original Vaswani transformer

#### 2.2 Learned Positional Embeddings
```python
pos_emb = nn.Embedding(max_seq_len, d_model)
```
- Learned during training (like GPT-2)
- Limited to max training sequence length
- Often performs better empirically

#### 2.3 RoPE (Rotary Position Embedding)
- Rotates query and key vectors based on position
- Relative position information
- Better for long contexts
- Used in Llama, PaLM

#### 2.4 ALiBi (Attention with Linear Biases)
- Adds bias to attention scores based on distance
- No positional embeddings needed
- Excellent extrapolation to longer sequences

### Implementation
```python
# Learned positional encoding (GPT-2 style)
token_emb = nn.Embedding(vocab_size, n_embd)
pos_emb = nn.Embedding(block_size, n_embd)

# In forward pass
tok_emb = token_emb(tokens)  # (B, T, C)
pos = torch.arange(0, T, device=device)
pos_emb_out = pos_emb(pos)  # (T, C)
x = tok_emb + pos_emb_out    # (B, T, C) - broadcasting
```

---

## 3. Feed Forward Networks (FFN)

### Architecture
```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = F.gelu(self.fc1(x))     # (B, T, d_ff)
        x = self.dropout(x)
        x = self.fc2(x)              # (B, T, d_model)
        return x
```

### Key Points
- **Expansion**: Typically `d_ff = 4 * d_model` (4x expansion)
- **Applied token-wise**: Same FFN applied to each token independently
- **Most parameters**: FFN contains ~2/3 of transformer parameters
- **Residual connection**: `x = x + FFN(LayerNorm(x))`

### From Scratch (No nn.Linear)
```python
# Forward pass
h = x @ W1 + b1              # (B, T, d_model) @ (d_model, d_ff)
h = gelu(h)                  # activation
out = h @ W2 + b2            # (B, T, d_ff) @ (d_ff, d_model)

# Backward pass
dL/dW2 = h.T @ dout
dL/db2 = dout.sum(dim=0)
dh = dout @ W2.T
dh = dh * gelu_derivative(h)
dL/dW1 = x.T @ dh
dL/db1 = dh.sum(dim=0)
```

---

## 4. KV Cache

### The Problem
During autoregressive generation, we recompute attention for all previous tokens at each step - wasteful!

```python
# Without KV cache: recompute everything each time
for i in range(seq_len):
    # Recompute K, V for all previous tokens (inefficient!)
    K = W_k @ x[:i+1]
    V = W_v @ x[:i+1]
    Q = W_q @ x[i]
    out = attention(Q, K, V)
```

### The Solution
Cache the key and value matrices for previously processed tokens.

```python
# With KV cache
K_cache = []  # Store all previous keys
V_cache = []  # Store all previous values

for i in range(seq_len):
    # Only compute K, V for new token
    k_new = W_k @ x[i]
    v_new = W_v @ x[i]

    # Append to cache
    K_cache.append(k_new)
    V_cache.append(v_new)

    # Use full cache for attention
    K = torch.cat(K_cache, dim=1)  # All previous + current
    V = torch.cat(V_cache, dim=1)
    Q = W_q @ x[i]

    out = attention(Q, K, V)
```

### Implementation Details
```python
# Shape tracking
Q: (batch, num_heads, 1, head_dim)           # Only current token
K: (batch, num_heads, seq_len, head_dim)     # All tokens so far
V: (batch, num_heads, seq_len, head_dim)     # All tokens so far

# Attention scores
scores = Q @ K.transpose(-2, -1)  # (batch, num_heads, 1, seq_len)
```

### Memory Considerations
- **Memory**: `2 * n_layers * n_heads * head_dim * seq_len` per batch item
- **Trade-off**: Memory for speed
- **Multi-head**: Separate cache per attention head

---

## 5. GELU + SwiGLU from Scratch

### 5.1 GELU (Gaussian Error Linear Unit)

#### Mathematical Definition
```python
GELU(x) = x * Φ(x) = x * P(X ≤ x), where X ~ N(0,1)
```

#### Approximations

**Exact (with error function)**:
```python
def gelu_exact(x):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
```

**Tanh approximation** (faster):
```python
def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2/math.pi) * (x + 0.044715 * x**3)
    ))
```

#### Why GELU?
- Smooth, non-monotonic
- Better gradient flow than ReLU
- Used in BERT, GPT-2, GPT-3
- Allows small negative values (unlike ReLU)

#### Derivative
```python
def gelu_derivative(x):
    # For backpropagation
    cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    pdf = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    return cdf + x * pdf
```

### 5.2 SwiGLU (Swish-Gated Linear Unit)

#### Mathematical Definition
```python
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
where Swish(x) = x * sigmoid(x)
```

#### Implementation
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=True)
        self.V = nn.Linear(d_model, d_ff, bias=True)

    def forward(self, x):
        # x: (B, T, d_model)
        swish_output = self.swish(self.W(x))  # (B, T, d_ff)
        gate_output = self.V(x)                # (B, T, d_ff)
        return swish_output * gate_output      # Element-wise multiply

    def swish(self, x):
        return x * torch.sigmoid(x)
```

#### From Scratch (Manual)
```python
def swiglu_manual(x, W, V, b, c):
    """
    x: (B, T, d_model)
    W: (d_model, d_ff)
    V: (d_model, d_ff)
    """
    # Gate branch
    gate = x @ W + b              # (B, T, d_ff)
    gate = gate * torch.sigmoid(gate)  # Swish activation

    # Value branch
    value = x @ V + c             # (B, T, d_ff)

    # Combine
    return gate * value           # (B, T, d_ff)
```

#### Why SwiGLU?
- Used in PaLM, Llama
- Better performance than GELU in some settings
- Gating mechanism helps with information flow
- Typically needs 2/3 reduction in d_ff to match parameters

#### Comparison
```python
# Standard FFN with GELU
FFN(x) = W2 * GELU(W1 * x)

# FFN with SwiGLU
FFN(x) = W2 * SwiGLU(x, W_gate, W_value)

# Note: SwiGLU uses 3 weight matrices vs 2 for standard FFN
```

---

## 6. LoRA (Low-Rank Adaptation)

### Core Idea
Instead of fine-tuning all parameters, inject trainable low-rank matrices into frozen weights.

### Mathematical Foundation

**Standard fine-tuning**:
```python
h = W * x  # Update all of W
```

**LoRA**:
```python
h = W_frozen * x + (B * A) * x
# W_frozen: (d, k) - pretrained, frozen
# A: (r, k) - trainable
# B: (d, r) - trainable
# r << min(d, k)  (rank is much smaller)
```

### Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        # Frozen pretrained weight
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # Initialize A with random, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Original path (frozen)
        result = self.linear(x)

        # LoRA path (trainable)
        lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return result + lora_result
```

### Key Parameters

#### Rank (r)
- Typical values: 4, 8, 16, 32
- Lower rank = fewer parameters, faster training
- Higher rank = more expressiveness
- Rule of thumb: Start with 8

#### Alpha (α)
- Scaling factor: `scaling = α / r`
- Typical values: 8, 16, 32
- Higher alpha = stronger LoRA influence
- Often set to `α = 2 * r`

### Parameter Savings

**Example**: Fine-tuning 7B model
- Full fine-tuning: 7B trainable parameters
- LoRA (r=8): ~0.8M trainable parameters (~1000x reduction!)

```python
# Parameters calculation
full_params = d * k
lora_params = r * (d + k)
reduction = full_params / lora_params

# Example: d=4096, k=4096, r=8
full_params = 16,777,216
lora_params = 8 * 8,192 = 65,536
reduction = 256x
```

### Where to Apply LoRA?

**Common choices**:
1. **Attention only**: Q, K, V, O projections
2. **Attention + FFN**: All linear layers
3. **Q, V only**: Often sufficient (Llama approach)

```python
# Typical configuration
class TransformerBlockWithLoRA(nn.Module):
    def __init__(self, ...):
        # Apply LoRA to attention matrices
        self.q_proj = LoRALayer(d_model, d_model, rank=8)
        self.k_proj = LoRALayer(d_model, d_model, rank=8)
        self.v_proj = LoRALayer(d_model, d_model, rank=8)
        self.o_proj = LoRALayer(d_model, d_model, rank=8)
```

### Merging Weights

After training, you can merge LoRA weights into the base model:

```python
def merge_lora_weights(base_weight, lora_A, lora_B, alpha, rank):
    """
    Merge LoRA weights back into base model
    """
    scaling = alpha / rank
    merged_weight = base_weight + (lora_B @ lora_A) * scaling
    return merged_weight

# Now you have a single weight matrix - no inference overhead!
```

### Benefits
- **Memory efficient**: Only store/update small matrices
- **Fast training**: Fewer parameters to update
- **Modular**: Can train multiple LoRA adapters for different tasks
- **No inference overhead**: Can merge weights after training
- **Easy deployment**: Just ship LoRA weights (MBs vs GBs)

---

## Next Steps

Once you've mastered these basics:
1. Build a simple transformer from scratch using these components
2. Study the complete [REVISION_TOPICS.md](./REVISION_TOPICS.md)
3. Focus on Section 1 (Core Transformer Architecture)
4. Practice implementing each component without libraries

## Practice Resources

- **TensorGym**: Interactive exercises for these concepts
- **NanoGPT**: Minimal GPT implementation to study
- **Karpathy's Videos**: Neural Networks Zero to Hero series
- **Microsoft LoRA Implementation**: Reference implementation
