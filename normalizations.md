import torch
# Batch Norm: Norm by feature/channel/last-dim
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

# Toy Example of Batch Norm for "normalizing" different scales for smoother learning (MLP) -> concept applies for Transformers when we add to residuals

Layer 1 output: [100, 0.01, 50]  ← Wildly different scales!
              ↓
         [Weights W]
              ↓
Layer 2 input: [very large values]
```

**What happens during backpropagation?**

Let's trace gradients with simple numbers:

**Forward pass:**
```
x = [100, 0.01, 50]
W = [[0.1, 0.2],
     [0.3, 0.4],
     [0.5, 0.6]]
     
y = x @ W = [35.003, 50.006]
```

**Backward pass (computing ∂Loss/∂x):**
```
If ∂Loss/∂y = [1, 1]

Then: ∂Loss/∂x₁ = 1×0.1 + 1×0.2 = 0.3
      ∂Loss/∂x₂ = 1×0.3 + 1×0.4 = 0.7
      ∂Loss/∂x₃ = 1×0.5 + 1×0.6 = 1.1
```

**But here's the problem**: When you update x:
```
x₁ = 100 - lr×0.3    (small relative change: 0.3%)
x₂ = 0.01 - lr×0.7   (HUGE relative change: 7000%!)
x₃ = 50 - lr×1.1     (medium relative change: 2.2%)
```

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
x_out = [1.42, -2.82, 1.42]  (with γ=2, β=0)
```

Now all values are on similar scales, so gradients are balanced:
```
Updates are proportional:
- Each dimension changes by similar percentages
- No dimension explodes or vanishes
- Training is STABLE
```

## Visual Diagram
```
WITHOUT LAYERNORM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input scales:  [████████████ 100]
               [▏ 0.01]
               [██████ 50]
                    ↓
Gradient flow:  Chaotic, unstable
                    ↓
Result:        Some neurons die, others explode


WITH LAYERNORM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input scales:  [████████████ 100]
               [▏ 0.01]              } → LayerNorm
               [██████ 50]
                    ↓
Normalized:    [███ 0.71]
               [███ -1.41]
               [███ 0.71]
                    ↓
Gradient flow:  Smooth, balanced
                    ↓
Result:        All neurons train effectively
```

## Why This Matters Specifically in Transformers

Transformers have **residual connections** everywhere:
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
Layer 3:  x = [previous] + attention_output
...
Layer 12: x = [10000, -5000, 8000]  ← EXPLODED!
```

**With LayerNorm** (placed strategically):
```
Layer 1:  x = [1, 2, 3]
          → Attention → LayerNorm → [normalized values]
Layer 2:  Add residual → LayerNorm → [normalized values]
...
Layer 12: Still [reasonable scale]  ← STABLE!
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