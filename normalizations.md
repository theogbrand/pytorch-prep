import torch
# Batch Norm:
def batch_norm(data: torch.Tensor, epsilon) -> torch.Tensor:
    t_mean = torch.mean(data, dim=0, keepdim=True)
    t_var = torch.var(data,dim=0, keepdim=True) # or use unbiased=False to use N instead of N-1 as the denominator
    return (data-t_mean)/torch.sqrt(t_var-epsilon)

## Layer Norm (Standard in Transformers)
# For each sample independently, normalize across all features
data.shape = (N, C)  # N samples, C features

# For sample i:
mean = torch.mean([feature_1, feature_2, ..., feature_C])  # across features
var = torch.var([feature_1, feature_2, ..., feature_C])    # across features

**The key insight:** Batch norm uses "horizontal" statistics (across samples), Layer norm uses "vertical" statistics (across features within each sample).

**Why Transformers use Layer Norm:**
- **Variable sequence lengths**: batch norm breaks when sequences have different lengths
- **Small/dynamic batches**: batch norm needs decent batch sizes for stable statistics
- **Autoregressive generation**: at inference, you might process one token at a time (batch size = 1)

## RMSNorm (Modern Upgrade)

Used in LLaMA, Mistral, and other recent LLMs. It's a simplified Layer Norm:

```python
# Layer Norm: (x - mean) / std
# RMSNorm: x / rms  (where rms = sqrt(mean(x²)))
```

**The simplification:** Skips the mean centering step, just scales by root-mean-square.

**Why it's popular:**
- ~10-15% faster (one less operation)
- Empirically works just as well for LLMs
- Simpler gradient flow

## Quick Comparison Table

| Norm | Normalizes Across | Common Use Case |
|------|-------------------|-----------------|
| Batch Norm | Samples (batch dim) | CNNs, fixed-size inputs |
| Layer Norm | Features (within sample) | Transformers, NLP |
| RMSNorm | Features (simplified) | Modern LLMs |

**Practical note:** Almost every Transformer you'll encounter uses Layer Norm or RMSNorm. Batch norm is essentially absent from the Transformer world because it fundamentally conflicts with variable-length sequences and single-sample generation.

# Toy Example (MLP) -> concept applies for Transformers when we add to residuals
Core Concept: LN Rescales values to std=1 and mean=0 via classic "Standardization" we know and love from statistics.
This way, we ensure that parameter updates are proportional:
- Each dimension changes by similar percentages
- No dimension explodes or vanishes
- Training is STABLE

### Without LN:
With learning rate = 0.1:
- x₁: 100 → 99.97 (barely moves)
- x₂: 0.01 → -0.06 (explodes by 600%!)
- x₃: 50 → 49.89 (reasonable)

### With LayerNorm

**Forward pass:**
```
x = [100, 0.01, 50]
        ↓
   [LayerNorm]  ← Normalize to mean=0, std=1
        ↓
x_norm = [0.71, -1.41, 0.71]  ← Similar scales!
        ↓
   [× γ + β]
        ↓
x_out = [1.42, -2.82, 1.42]  (with γ=2, β=0) -> parameters of the LayerNorm
```

Now all values are on similar scales, so gradients are balanced:

## Why This Matters Specifically in Transformers

Transformers have **residual connections**, where we add the output of a layer to the input everywhere:
```
      x ────────────────┐
      │                 │
   [Attention]          │
      │                 │
   [LayerNorm]          │
      │                 │
      └────── + ←───────┘  ← Residual connection
```

**Without LayerNorm**, after many layers:
```
Layer 1:  x = [1, 2, 3]
Layer 2:  x = [1, 2, 3] + attention_output
Layer 3:  x = [Layer 2 output] + attention_output
...
Layer 12: x = [10000, -5000, 8000]  ← EXPLODED!
```

**With LayerNorm** (placed strategically):
```
Layer 1:  x = [1, 2, 3]
          → Attention → LayerNorm → [normalized values]
Layer 2:  Add residual → LayerNorm → [normalized values]
...
Layer 12: Still [increases but at reasonable scale, especially important for deep networks]  ← STABLE!
```

## Concrete Training Benefit

Let me show you actual gradient magnitudes:

### Without LayerNorm (deep network):
```
Layer 12 gradient: 0.0001
Layer 6 gradient:  0.01
Layer 1 gradient:  0.00000001  ← Vanishing!

Can't use high learning rates → slow training
```

### With LayerNorm:
```
Layer 12 gradient: 0.1
Layer 6 gradient:  0.1
Layer 1 gradient:  0.1  ← All similar!

Can use learning rate 10x higher → fast training

# Drawbacks of LayerNorm:
## Drawback 1: Information Loss, hard to learn certain patterns
for 2 very different inputs, after LN they look the same
Input A: [100, 101, 102]  ← High values, small variance
Input B: [1, 50, 99]      ← Medium values, large variance

**After LayerNorm:**
```
Input A normalized:
  mean = 101, std = 0.82
  result: [-1.22, 0, 1.22]

Input B normalized:
  mean = 50, std = 40.1
  result: [-1.22, 0, 1.22]
```

**They become IDENTICAL!** 🚨

The network completely loses:
- The **scale** information (A was 100× larger than B)
- The **spread** information (B was 50× more variable)
- The **absolute position** (A was centered at 101, B at 50)

## Drawback 2: Breaks with Variable Sequence Lengths
(more relevant for RNNs)

## Drawback 3: Computational Overhead (Computing and storing in memory)
With LN:     Store x, mean, variance, normalized_x (for backprop)


Important variants of LN:
1.	Post-LN (original Transformer, 2017)
y = LN(x + Attn(x))
z = LN(y + MLP(y))

2.	Pre-LN (GPT-style, most modern LLMs)
y = x + Attn(LN(x))
z = y + MLP(LN(y))

RESIDUAL CONNECTION:                                       │
│  • Provides gradient highway (backward pass)                │
│  • Preserves information (forward pass)                     │
│  • Problem: Can cause scale explosion                       │
│                                                             │
│  LAYERNORM:                                                 │
│  • Controls scale explosion (forward pass)                  │
│  • Stabilizes training (both directions)                    │
│  • Problem: Alone would hurt gradient flow 