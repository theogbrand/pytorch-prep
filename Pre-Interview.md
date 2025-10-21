PyTorch Round 1:

1. Attention Block
    - Masked, Sliding Window, ALiBi (https://leetgpu.com/challenges/attention-with-linear-biases)
    - KV Cache Mechanics (https://www.deep-ml.com/deep-0/qg_107)
    - [Positional Encoding](positional_encodings.py)
    - Multi-Head Attention
        - n_dim = num_heads * head_size (Then head_size is determined last after n_dim and num_heads, by dividing n_dim by the number of heads)
    - Residual Connections: Add the input to the output of the attention block (x = x + MHA(x); x = x + FFN(x))
        - The projection layer and residual connection work together but serve different purposes - the projection transforms the representation while the residual connection helps with gradient flow and feature preservation.
    - Activation Functions: 
        - SwiGLU
            ```python
            x1, x2 = input[:N//2], input[N//2:] # splits the input into two halves
            s1 = x1 * torch.sigmoid(x1)
            output = torch.mul(s1, x2, out=output)
            ```
2. SoftMax
    - Forward and Backward Pass (see CEL Makemore)
    - Happens in CEL, Attention Block
3. Cross Entropy Loss/NLL
    - Cross Entropy Loss is essentially: -log(softmax(logits))[correct_indices].mean()
    - Perplexity is just e^CELoss
    - Clipping: torch.clip(probabilities, min=epsilon, max=1.0) to avoid log(0)
    - When given logits:
    ```python
        logits_maxes = logits.max(dim=1, keepdim=True).values 
        norm_logits = logits - logits_maxes 
        counts = norm_logits.exp()
        counts_sum = counts.sum(dim=1, keepdims=True)
        counts_sum_inv = counts_sum**-1 
        probs = counts * counts_sum_inv
        logprobs = probs.log()
        loss = -logprobs[range(batch_size), Yb].mean()
    ```
    - When given probabilities:
    ```python
        clipped_prob = torch.clip(probabilities, min=epsilon, max=1.0)
        loss = -torch.log(clipped_prob[range(batch_size), Yb]).mean()
    ```
3. LayerNorm (Pre/Post) v.s BatchNorm v.s RMSNorm
    - BatchNorm for 2D tensor (B,C) input tensor; for large batch sizes >= 32, CV CNNs; Normalize ↓
    ```python
        mean_t = torch.mean(input, dim=0, keepdim=True) # Batch Dim is 0
        sigma2_t = torch.pow(input - mean_t, 2).mean()
        x_norm_t = (input - mean_t) / torch.sqrt(sigma2_t + epsilon)
        return output.copy_(gamma * x_norm_t + beta)
    ```
    - LayerNorm for 2D tensor (B,T,C) input tensor; NLP, different sample have different distributions, same behavior in training and inference; Normalize →
    ```python
        mean_t = torch.mean(input, dim=-1, keepdim=True)
        sigma2_t = torch.var(input, dim=-1, keepdim=True, unbiased=True) # Kaparthy and Bessel's Correction are unbiased!!!
        x_norm_t = (input - mean_t) / torch.sqrt(sigma2_t + epsilon)
        return output.copy_(x_norm_t)
    ```
    - Layer/BatchNorm1D -> Sequences or 0D (B,C); BatchNorm2D -> Images; BatchNorm3D -> Videos
    - If asked to BatchNorm over Batch and Spatial Dimensions for 4D Tensor (B,C,H,W):
        ```python
            mean_t = input.mean(dim=(0,2,3), keepdim=True)
            sigma2_t = input.var(dim=(0,2,3), keepdim=True, unbiased=True)
            x_norm_t = (input - mean_t) / torch.sqrt(sigma2_t + epsilon)
            return output.copy_(gamma * x_norm_t + beta)
        ```
        - equivalent to: output_builtin = nn.BatchNorm2d(x)
4. Backward Pass Rules (Addition, Multiplication, Subtraction)
    - Common activation functions (ReLU, Sigmoid, Tanh, SwiGLU, SILU) derivatives
5. CE/BCE Loss, MSE Loss
    - Manual backward pass and computing loss per epoch for a simple 2-Layer MLP
6. Count parameters in a 2-Layer MLP
    - for parameter in model.parameters(): total_params += parameter.numel()
7. Tokenization BPE
8. Transformers Vaswani Architecture (encoder-decoder architecture)
9. GPT2 Architecture (decoder-only architecture)
10. Broadcasting Rules for Tensors v.s Matrices
    - Tensor BT, alights right first, then "add ones" on both, then BT over 1-dims ONLY
    - Matrix BT, aligns LEFT first, BT over batch dims ONLY, and then check that matmul is valid with matching inner dims 
11. Dot Product v.s Matmul
12. Important Torch APIs:
    - torch.mean()
        - mean_t = torch.mean(input, dim=0, keepdim=True) -> Batch Norm for [B,C] input tensor
        - When averaging heads, keepdim=False (we want to squeeze):     return torch.mean(multi_head_output, dim=1, keepdim=False)
        - keepdim=True is an automatic unsqueeze(index_to_add_1dim); if no index provided, removes all size-1 dimensions
            - Important: squeeze only works on size-1 dimensions! nothing happens if squeeze a non-size-1 dimension
    - torch.round()
    - torch.exp()
    - torch.log()
    - torch.sqrt()
    - torch.pow()
        - BatchNorm Variance: sigma2_t = torch.pow(input - mean_t, 2).mean()
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