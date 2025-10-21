PyTorch Round 1:

1. Attention Block
    - Masked, Sliding Window, ALiBi (https://leetgpu.com/challenges/attention-with-linear-biases)
    - KV Cache Mechanics
2. SoftMax
    - Forward and Backward Pass (see CEL Makemore)
    - Cross Entropy Loss is essentially: -log(softmax(logits))[correct_indices].mean()
3. LayerNorm (Pre/Post) v.s BatchNorm v.s RMSNorm
4. Backward Pass Rules (Addition, Multiplication, Subtraction)
    - Common activation functions (ReLU, Sigmoid, Tanh, SwiGLU, SILU) derivatives
5. CE/BCE Loss, MSE Loss
    - Manual backward pass and computing loss per epoch for a simple 2-Layer MLP
6. Count parameters in a 2-Layer MLP
7. Tokenization BPE
8. Transformers Vaswani Architecture (encoder-decoder architecture)
9. GPT2 Architecture (decoder-only architecture)
10. Broadcasting Rules for Tensors v.s Matrices
11. Dot Product v.s Matmul
12. Important Torch APIs:
    - torch.mean()
        - When averaging heads, keepdim=False (we want to squeeze):     return torch.mean(multi_head_output, dim=1, keepdim=False)
        - keepdim=True is an automatic unsqueeze(index_to_add_1dim); if no index provided, removes all size-1 dimensions
            - Important: squeeze only works on size-1 dimensions! nothing happens if squeeze a non-size-1 dimension
    - torch.round()
    - torch.exp()
    - torch.log()
    - torch.sqrt()
    - torch.pow()
    - torch.mul()
    - torch.div()
    - torch.add()
    - torch.sub()
    - torch.matmul()
    - torch.tril()
    - torch.as_tensor()
    - torch.dot()
    - torch.where()
    - torch.sum()
        - in softmax: denom = torch.sum(exp, dim=1, keepdim=True) # shinks across T dim
    - torch.max()
        - in softmax: max_t = torch.max(input, dim=1, keepdim=True).values # shrink across dim
            - Returns keys values + indices
            - max/sum across the "Time/Sequence" dimension which is dim=1 for Batched Input (B,T,C) or dim=0 for non_batched input (rows, columns)
            - keepdim=True especially for downstream use!!!
    - torch.argmax()
    - torch.arange()
13. Important Torch Functions:
    - F.softmax()