"""
Positional Encodings in Transformers
=====================================
Common use cases for torch.arange() in Transformer architectures.
"""

import torch
import torch.nn as nn
import math


def pos_encoding(position: int, d_model: int): # simple
    
    if position == 0 or d_model <= 0:
        return -1

    # Create position and dimension indices
    pos = torch.arange(position, dtype=torch.float32).reshape(position, 1)
    ind = torch.arange(d_model, dtype=torch.float32).reshape(1, d_model)

    # Compute the angles
    angle_rads = pos / torch.pow(10000, (2 * (ind // 2)) / d_model)

    # Apply sine to even indices, cosine to odd indices
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # Even indices
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # Odd indices

    # Convert to float16
    return angle_rads.to(torch.float16)

# ==============================================================================
# 1. SINUSOIDAL POSITIONAL ENCODING (Original Transformer - Vaswani et al.)
# ==============================================================================

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Creates fixed sinusoidal positional encodings from "Attention is All You Need".
    
    Why this matters:
    - Transformers have no inherent notion of position (permutation invariant)
    - We need to inject positional information into the model
    - Sinusoidal allows extrapolation to longer sequences than seen during training
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model)) # even pos
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) # odd pos
    
    Args:
        seq_len: Length of the sequence (e.g., number of tokens)
        d_model: Model dimension (must be even)
    
    Returns:
        Positional encoding tensor of shape (seq_len, d_model)
    """
    # Step 1: Create position indices [0, 1, 2, ..., seq_len-1]
    # Shape: (seq_len,) -> (seq_len, 1) for broadcasting
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    
    # Step 2: Create dimension indices and compute div_term
    # For i in [0, 2, 4, ..., d_model-2], compute 1/10000^(i/d_model)
    # Mathematical equivalence (log/exp trick):
    #   1/10000^(i/d_model) = 10000^(-i/d_model)
    #                       = e^(ln(10000^(-i/d_model)))    [since a = e^(ln(a))]
    #                       = e^((-i/d_model) * ln(10000))  [since ln(a^b) = b*ln(a)]
    # NOTE: exp/log is more numerically stable than torch.pow(10000, i/d_model) because:
    #   - Avoids overflow/underflow from large powers (10000^x can be huge or tiny)
    #   - exp operates in log-space where numbers stay in reasonable ranges
    #   - Reduces floating-point precision errors from repeated multiplications
    i = torch.arange(0, d_model, 2, dtype=torch.float)  # [0, 2, 4, ...]
    exponent = -(i / d_model) * math.log(10000.0)      # -(i/d_model) * ln(10000)
    div_term = torch.exp(exponent)                      # e^(-(i/d_model) * ln(10000))
    # Shape: (d_model/2,)
    
    # Step 3: Initialize PE matrix
    pe = torch.zeros(seq_len, d_model)
    
    # Step 4: Apply sin to even indices (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Step 5: Apply cos to odd indices (1, 3, 5, ...)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Module version with dropout, ready for production use."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encodings
        pe = sinusoidal_positional_encoding(max_len, d_model)
        # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ==============================================================================
# 2. CAUSAL ATTENTION MASK (GPT-style autoregressive models)
# ==============================================================================

def create_causal_mask(seq_len):
    """
    Creates causal (autoregressive) attention mask for decoder-only models.
    
    Why this matters:
    - In GPT-style models, token at position i can only attend to positions 0...i
    - This prevents information leakage from future tokens during training
    - Essential for autoregressive generation
    
    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = masked (blocked)
    """
    # Method 1: Using torch.arange() for explicit position comparison
    # Create query positions: [[0], [1], [2], ..., [seq_len-1]]
    i = torch.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    
    # Create key positions: [[0, 1, 2, ..., seq_len-1]]
    j = torch.arange(seq_len)[None, :]  # Shape: (1, seq_len)
    
    # Token i should NOT attend to token j if j > i
    # Broadcasting: (seq_len, 1) vs (1, seq_len) -> (seq_len, seq_len)
    mask = j > i
    
    return mask


def create_causal_mask_triu(seq_len):
    """
    Alternative: Using torch.triu() - more common in practice.
    
    Returns upper triangular matrix (above diagonal = True = masked).
    """
    # diagonal=1 means the first diagonal above the main diagonal
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


# ==============================================================================
# 3. RELATIVE POSITION ENCODING (T5-style, Transformer-XL)
# ==============================================================================

def relative_position_matrix(seq_len):
    """
    Creates relative position matrix: rel_pos[i,j] = j - i
    
    Why this matters:
    - Instead of absolute positions, encode relative distances
    - More generalizable to variable length sequences
    - Used in T5, Transformer-XL, and many modern architectures
    
    Returns:
        Relative position matrix of shape (seq_len, seq_len)
        Entry [i,j] = position_j - position_i
    """
    # Query positions: [[0], [1], [2], ...]
    q_pos = torch.arange(seq_len)[:, None]  # (seq_len, 1)
    
    # Key positions: [[0, 1, 2, ...]]
    k_pos = torch.arange(seq_len)[None, :]  # (1, seq_len)
    
    # Relative positions: how far is key from query?
    # Positive = key is after query, Negative = key is before query
    rel_pos = k_pos - q_pos
    
    return rel_pos


def relative_position_bucket(relative_positions, num_buckets=32, max_distance=128):
    """
    Buckets relative positions (T5 approach).
    
    Why bucketing?
    - Exact relative positions become sparse for large distances
    - Bucket nearby positions finely, distant positions coarsely
    - Reduces parameter count while maintaining expressiveness
    
    T5 uses:
    - Half buckets for negative positions (key before query)
    - Half buckets for positive positions (key after query)
    - Logarithmic bucketing for distances beyond threshold
    """
    num_buckets //= 2
    ret = (relative_positions > 0).long() * num_buckets
    n = torch.abs(relative_positions)
    
    # Half of buckets for exact positions (0, 1, 2, ...)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    
    # Other half for logarithmically spaced positions
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / 
        math.log(max_distance / max_exact) * 
        (num_buckets - max_exact)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    
    ret = ret + torch.where(is_small, n, val_if_large)
    return ret


# ==============================================================================
# 4. LEARNED POSITIONAL EMBEDDINGS (BERT, GPT-2)
# ==============================================================================

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (like BERT/GPT-2).
    
    Why learned vs sinusoidal?
    - More flexible, can learn task-specific positional patterns
    - Can't extrapolate beyond max_len seen during training
    - Often performs better for fixed-length tasks
    """
    
    def __init__(self, max_len, d_model):
        super().__init__()
        # Create learnable embedding table
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)
        
        # Look up embeddings: (seq_len,) -> (seq_len, d_model)
        pos_emb = self.pos_embedding(positions)
        
        # Add to input (broadcasting over batch)
        return x + pos_emb.unsqueeze(0)


# ==============================================================================
# 5. ROTARY POSITIONAL EMBEDDING (RoPE) - Used in LLaMA, GPT-NeoX
# ==============================================================================

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    """
    Precompute rotation frequencies for RoPE (Rotary Position Embedding).
    
    Why RoPE?
    - Encodes relative position information through rotation
    - Linear attention has relative position bias property
    - Better extrapolation to longer sequences
    - Used in LLaMA, GPT-J, GPT-NeoX, PaLM
    
    Key insight: Rotate embedding space by position-dependent angle
    """
    # Step 1: Compute frequencies for each dimension pair
    # theta^(-2i/d) for i in [0, 1, ..., dim/2-1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Step 2: Create position indices
    t = torch.arange(seq_len, dtype=torch.float)
    
    # Step 3: Outer product: positions × frequencies
    # Shape: (seq_len, dim/2)
    freqs = torch.outer(t, freqs)
    
    # Step 4: Convert to complex numbers (cos + i*sin)
    # This represents rotation in 2D space
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary embeddings to query or key tensors.
    
    Args:
        x: Tensor of shape (..., seq_len, dim)
        freqs_cis: Complex frequencies of shape (seq_len, dim/2)
    """
    # Reshape x to (..., seq_len, dim/2, 2) and view as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Multiply by rotation (complex multiplication = rotation)
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    
    return x_out.type_as(x)


# ==============================================================================
# 6. ALiBi - Attention with Linear Biases (Used in BLOOM, MPT)
# ==============================================================================

def get_alibi_slopes(num_heads):
    """
    Get ALiBi slopes for each attention head.
    
    Why ALiBi?
    - No positional embeddings needed at all!
    - Add biases directly to attention scores
    - Excellent extrapolation to longer sequences
    - Memory efficient (no extra embeddings)
    """
    def get_slopes(n):
        # Geometric sequence: 2^(-8/n), 2^(-2*8/n), ...
        start = 2 ** (-8 / n)
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    # For non-power-of-2 heads, use closest power of 2
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    slopes = torch.tensor(get_slopes(closest_power_of_2))
    
    if closest_power_of_2 != num_heads:
        extra = get_slopes(2 * closest_power_of_2)
        slopes = torch.cat([slopes, torch.tensor(extra[::2][:num_heads - closest_power_of_2])])
    
    return slopes


def create_alibi_bias(seq_len, num_heads):
    """
    Create ALiBi attention bias matrix.
    
    Returns:
        Bias tensor of shape (num_heads, seq_len, seq_len)
    """
    # Relative position matrix using torch.arange
    q_pos = torch.arange(seq_len)[:, None]
    k_pos = torch.arange(seq_len)[None, :]
    rel_pos = k_pos - q_pos  # (seq_len, seq_len)
    
    # Get slopes for each head
    slopes = get_alibi_slopes(num_heads)  # (num_heads,)
    
    # Multiply: (num_heads, 1, 1) * (1, seq_len, seq_len)
    alibi = slopes[:, None, None] * rel_pos[None, :, :]
    
    return alibi


# ==============================================================================
# INTERVIEW PREP: KEY QUESTIONS & ANSWERS
# ==============================================================================

"""
Q1: Why do we need positional encodings at all?
A: Transformers are permutation-invariant (self-attention has no position info).
   Without PE, "I love Paris" = "Paris love I"

Q2: Sinusoidal vs Learned - which is better?
A: Learned: Better for fixed-length tasks (BERT)
   Sinusoidal: Better extrapolation to unseen lengths (original Transformer)
   RoPE/ALiBi: Best of both worlds (LLaMA, BLOOM)

Q3: What's the advantage of relative positions over absolute?
A: Generalization - "word 2 tokens ago" is more meaningful than "word at position 573"
   T5 showed 3B model with relative PE outperforms absolute PE

Q4: How does RoPE differ from sinusoidal?
A: Applied to Q/K directly via rotation, not added to embeddings
   Naturally encodes relative position in attention scores
   Better length extrapolation

Q5: Why is ALiBi gaining popularity?
A: No embeddings needed = memory efficient
   State-of-the-art extrapolation
   Used in BLOOM (176B), MPT, Falcon

Q6: How to choose for a new model?
A: Short sequences (<512): Learned embeddings
   Long sequences: RoPE or ALiBi
   Variable length: RoPE or ALiBi
   Training efficiency: ALiBi (no extra params)
"""


if __name__ == "__main__":
    # Demo all approaches
    seq_len, d_model, num_heads = 10, 64, 8
    
    print("=" * 70)
    print("1. SINUSOIDAL POSITIONAL ENCODING")
    print("=" * 70)
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    print(f"Shape: {pe.shape}")
    print(f"First position encoding:\n{pe[0, :8]}")
    
    print("\n" + "=" * 70)
    print("2. CAUSAL MASK")
    print("=" * 70)
    mask = create_causal_mask(seq_len)
    print(f"Shape: {mask.shape}")
    print(f"Mask (True=blocked):\n{mask.int()}")
    
    print("\n" + "=" * 70)
    print("3. RELATIVE POSITION MATRIX")
    print("=" * 70)
    rel_pos = relative_position_matrix(seq_len)
    print(f"Shape: {rel_pos.shape}")
    print(f"Matrix:\n{rel_pos}")
    
    print("\n" + "=" * 70)
    print("4. ROPE FREQUENCIES")
    print("=" * 70)
    freqs = precompute_freqs_cis(d_model, seq_len)
    print(f"Shape: {freqs.shape}")
    print(f"First 3 positions, first 4 dims:\n{freqs[:3, :4]}")
    
    print("\n" + "=" * 70)
    print("5. ALiBi BIAS")
    print("=" * 70)
    alibi = create_alibi_bias(seq_len, num_heads)
    print(f"Shape: {alibi.shape}")
    print(f"Head 0 bias:\n{alibi[0]}")
    print(f"Slopes: {get_alibi_slopes(num_heads)}")

