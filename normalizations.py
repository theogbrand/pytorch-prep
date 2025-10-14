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

