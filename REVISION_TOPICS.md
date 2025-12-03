# LLM Research Scientist Interview Revision Guide

A comprehensive, structured guide covering all essential topics for LLM research scientist interviews. This guide synthesizes topics into clear categories with step-by-step coverage.

---

## 1. Core Transformer Architecture

### 1.1 Transformer Blocks
- **Vaswani Architecture** (encoder-decoder)
- **GPT2 Architecture** (decoder-only)
- **Llama Architecture**
- Debugging transformer blocks from scratch
- Tensor shape debugging

### 1.2 Attention Mechanisms
- **Self-Attention & Cross-Attention**
- **Multi-Head Attention (MHA)**
  - Head size calculation: `n_dim = num_heads * head_size`
  - Concatenation along channel dimension
- **Masked/Causal Attention**
- **Quadratic Problem** and long context challenges

### 1.3 Positional Encoding
- **Sin/Cos Positional Encoding**
- **Learned Positional Encoding**
- **RoPE (Rotary Position Embedding)**
- **YaRN** (long context extension)
- **ALiBi (Attention with Linear Biases)**

### 1.4 Residual Connections
- Add input to output: `x = x + MHA(x); x = x + FFN(x)`
- Projection layers with residual connections
- Gradient flow and feature preservation

### 1.5 Embeddings
- Token embeddings: `nn.Embedding(vocab_size, n_embd)`
- Position embeddings: `nn.Embedding(block_size, n_embd)`

---

## 2. Attention Optimizations

### 2.1 Efficient Attention
- **Sliding Window Attention**
  - Mask requires distance between tokens
  - Window size operationalization
- **Flash Attention**
- **KV Cache Mechanics**
  - Building KV cache matrices
  - Memory optimization for inference

### 2.2 Long Context Handling
- RoPE/YaRN implementations
- ALiBi implementation
- Continuous batching for efficient inference

**Resources:**
- [Continuous Batching Guide](https://huggingface.co/blog/continuous_batching)

---

## 3. Neural Network Fundamentals

### 3.1 Activation Functions
- **ReLU**: `torch.where(x > 0, x, zeros)`
- **Leaky ReLU**: `torch.where(x > 0, x, alpha * x)`
- **GELU**: `x * 0.5 * (1 + torch.erf(x / sqrt(2)))`
- **SwiGLU**: Split, apply swish, and gate
- Derivatives for backpropagation

### 3.2 Feedforward Networks (FFN)
- **FFN with Residual Connections and Dropout**
- Single neuron forward + backward pass
- FFN forward + backward pass
- Manual implementation without nn.Linear

### 3.3 Dropout
- **Forward Pass**: Binary mask with inverted dropout scaling
- **Backward Pass**: Apply same mask to gradients
- NumPy: `np.random.binomial(1, 1-p, size=x.shape)`
- PyTorch: `torch.bernoulli(torch.ones(x.shape) * (1-p))`

### 3.4 Backpropagation
- **Chain rule** application
- **Auto-differentiation**
- Addition, multiplication, subtraction operations
- Single neuron backprop
- Common activation function derivatives
- Custom autograd implementations

---

## 4. Normalizations

### 4.1 LayerNorm
- Normalize across feature dimension (→)
- Used in NLP, same behavior in training/inference
- Pre-norm vs Post-norm
- For (B,T,C): `mean/var over dim=-1`

### 4.2 BatchNorm
- Normalize across batch dimension (↓)
- For large batch sizes (≥32), used in CNNs
- BatchNorm1D (sequences), BatchNorm2D (images), BatchNorm3D (videos)
- For (B,C): `mean/var over dim=0`
- For (B,C,H,W): `mean/var over dim=(0,2,3)`

### 4.3 RMSNorm
- Root Mean Square normalization
- `sqrt(1/N * sum(input² + eps))`
- More efficient than LayerNorm

---

## 5. Loss Functions & Metrics

### 5.1 Softmax
- Forward and backward pass implementation
- **In Cross-Entropy**: sum over vocab/class dimension
- **In Attention**: sum over keys dimension (last dim)
- **In Generation**: sum over vocab dimension
- Numerical stability with log-sum-exp trick

### 5.2 Cross-Entropy Loss / NLL
- `CE = -log(softmax(logits))[correct_indices].mean()`
- From logits vs from probabilities
- Clipping to avoid log(0)
- Perplexity: `e^(CE Loss)`

### 5.3 Other Losses
- **Binary Cross-Entropy (BCE)**
- **Mean Squared Error (MSE)**
- Manual backward pass for simple MLPs

---

## 6. Optimizers & Training

### 6.1 Optimizers
- **AdamW** (Adam with weight decay)
  - First moment (m), second moment (v)
  - Bias correction
- **RMSProp**
- **SGD**
- **AdaGrad, AdaDelta**
- **Muon Optimizer**

### 6.2 Training Techniques
- **Distributed Training** implementations
- **Mixed Precision Training** (FP16/FP32)
- Learning rate schedulers
- Gradient accumulation
- Gradient clipping

### 6.3 Debugging & Best Practices
- Estimate train/val loss based on batch
- Parameter counting: `sum(p.numel() for p in model.parameters())`
- Tensor shape debugging
- Indexing error debugging

---

## 7. Tokenization

### 7.1 Byte Pair Encoding (BPE)
- Implementation from scratch
- Vocabulary building
- Encoding/decoding sequences

---

## 8. Advanced Architectures

### 8.1 Vision Transformers (ViT)
- Patch embeddings
- CLS token
- Image + text token handling
- `<image_pad>` placeholder tokens
- Vision-language integration
- **SIGLIP** implementation

**Resources:**
- [Umar Jamil ViT Tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s)
- [Image Transformer Paper](https://arxiv.org/abs/1802.05751)

### 8.2 Vision Language Models (VLMs)
- Multimodal transformers
- Image-text concatenation vs addition
- Marker tokens (`<vision_start>`, `<vision_end>`)
- Dynamic vs fixed image tokens
- Compressed tokens (Q-Former, BLIP-2)

### 8.3 Vision Language Action Models (VLAs)
- Action space integration
- Embodied AI applications

### 8.4 RNNs & LSTMs (Optional)
- Historical context: RNNs → LSTMs → Transformers
- Sequence modeling fundamentals

---

## 9. Mixture of Experts (MoE)

### 9.1 Core Concepts
- Expert routing
- Gating mechanisms
- Load balancing
- Sparse activation

### 9.2 Implementation
- **Top-K Selection** for expert routing
- Conditional computation
- Scaling considerations

**Resources:**
- [SeeMOE Guide](https://huggingface.co/blog/AviSoori1x/makemoe2)

---

## 10. Efficient Fine-Tuning

### 10.1 LoRA (Low-Rank Adaptation)
- Low-rank matrix decomposition
- Efficient parameter updates
- Rank selection
- Thinking Machine's LoRa without Regret [Impl](https://github.com/michaelbzhu/lora-without-regret)

**Resources:**
- [Microsoft LoRA Implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
- [TensorGym LoRA Exercise](https://tensorgym.com/exercises/17)

### 10.2 Quantization
- FP16, FP32 considerations
- Post-training quantization
- Quantization-aware training

---

## 11. Inference & Performance

### 11.1 Decoding Strategies
- **Top-K sampling**
- **Top-p (nucleus) sampling**
- **KNN-based methods**
- **Speculative Decoding**
- Binary search, backtracking
- Best-of-N (BoN) sampling with heap data structures

### 11.2 Generation
- Multinomial sampling: `torch.multinomial(probs, num_samples=1)`
- Temperature scaling
- Context window management: `idx_cond = idx[:, -block_size:]`

### 11.3 GPU Profiling
- Performance bottleneck identification
- Memory optimization
- Throughput optimization

---

## 12. PyTorch APIs & Functions

### 12.1 Essential Torch Operations
- **Statistics**: `torch.mean()`, `torch.var()`, `torch.sum()`
- **Math**: `torch.exp()`, `torch.log()`, `torch.sqrt()`, `torch.pow()`
- **Matrix ops**: `torch.matmul()`, `torch.dot()`, matrix transpose
- **Indexing**: `torch.arange()`, `torch.where()`
- **Aggregation**: `torch.max()`, `torch.argmax()`, `torch.min()`
- **Masking**: `torch.tril()`, `torch.triu()`
- **Utilities**: `torch.as_tensor()`, `torch.round()`, `torch.cat()`

### 12.2 Important Patterns
- `keepdim=True` for broadcasting
- `.values` and `.indices` from `torch.max()`
- `dim=-1` for channel/feature dimension
- `dim=0` for batch dimension
- `F.linear()` vs `nn.Linear()` weight transpose handling

### 12.3 Broadcasting Rules
- **Tensor Broadcasting**: Align right, add ones left, broadcast over 1-dims
- **Matrix Broadcasting**: Align left, broadcast over batch dims, check inner dims for matmul
- Dot product vs matmul behavior

---

## 13. Advanced Topics (Time Permitting)

### 13.1 Reinforcement Learning
- RLHF (Reinforcement Learning from Human Feedback)
- Policy optimization
- Reward modeling

#### RL Fundamentals
- MDP
- Deep Q Learning (DQN)
- AlphaGo + AlphaZero + AlphaGoZero + AlphaStar + MuZero
- AlphaFold
- "Asynchronous Methods for Deep Reinforcement Learning" (A3C)

### 13.2 Project Knowledge
- **VL-PRM** (Vision-Language Process Reward Model)

---

## Key References

- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [LucidRains Transformers](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py)
- [Stanford CS231N](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)
- [Stanford CS336](https://www.youtube.com/watch?v=6OBtO9niT00&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=7)

---

## Study Priority

**Fundamental (Must Know)**
1. Core Transformer Architecture (Sections 1, 3, 4, 5)
2. Attention Mechanisms & Optimizations (Section 2)
3. PyTorch APIs & Backpropagation (Sections 12, 3.4)
4. Tokenization (Section 7)

**Intermediate (Should Know)**
5. Optimizers & Training (Section 6)
6. Vision Transformers & VLMs (Section 8)
7. Inference & Decoding (Section 11)
8. LoRA & Efficient Fine-tuning (Section 10)

**Advanced (Good to Know)**
9. MoE (Section 9)
10. GPU Profiling (Section 11.3)
11. Reinforcement Learning (Section 13)
