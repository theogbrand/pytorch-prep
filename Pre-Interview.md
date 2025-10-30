PyTorch Round 1:

1. Attention Block
    - Masked/Causal (https://www.deep-ml.com/problems/107)
    - Sliding Window
    - Flash Attention
    - ALiBi (https://leetgpu.com/challenges/attention-with-linear-biases)
    - KV Cache Mechanics (https://www.deep-ml.com/deep-0/qg_107)
    - [Positional Encoding](positional_encodings.py)
    - Multi-Head Attention
        - n_dim = num_heads * head_size (Then head_size is determined last after n_dim and num_heads, by dividing n_dim by the number of heads)
    - Residual Connections: Add the input to the output of the attention block (x = x + MHA(x); x = x + FFN(x))
        - The projection layer and residual connection work together but serve different purposes - the projection transforms the representation while the residual connection helps with gradient flow and feature preservation.
        - [FFN with Residual and Dropout](https://www.deep-ml.com/problems/178)
            - Be clear to ask if 1) Residual connection BEFORE or AFTER dropout; 2) torch.round(x, decimals=4) OK or use torch.round(out * 10000) / 10000
            - The trick is to save ```residual = x``` in the first line before any computations are done
    - Activation Functions: 
        - SwiGLU
            ```python
            x1, x2 = input[:N//2], input[N//2:] # splits the input into two halves
            s1 = x1 * torch.sigmoid(x1)
            output = torch.mul(s1, x2, out=output)
            ```
    - Dropout Layers (FP + BP)
        - [Forward/Backward Pass Implementation](https://www.deep-ml.com/problems/151)
2. SoftMax
    - Know how to compute Forward and Backward Pass from scratch (see CEL Makemore) - know sum over which dim
    - Happens in:
        a) CEL: F.cross_entropy(logits, targets)
            - sum over vocab dim (classes), not time. dim=1 for B,T,C
        b) Attention Block: wei = F.softmax(wei, dim=-1)  # (B, T, T)
            - sum over keys dim, which is the last dim (recall what Q.K.T outputs -> the keys to attend to)
        c) Generation/Sampling: F.softmax(logits, dim=-1) (B, C); idx_next = torch.multinomial(probs, num_samples=1) (B, 1)
            - sum over vocab dim since logits shape (B, vocab_size)
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
    - [Single Neuron Backprop](https://www.deep-ml.com/problems/25)
5. CE/BCE Loss, MSE Loss
    - Manual backward pass and computing loss per epoch for a simple 2-Layer MLP
6. Count parameters in a 2-Layer MLP
    - for parameter in model.parameters(): total_params += parameter.numel()
7. Tokenization BPE
8. Transformers Vaswani Architecture (encoder-decoder architecture)
9. GPT2 Architecture (decoder-only architecture)
10. Broadcasting Rules for Tensors v.s Matrices
    - Tensor BT, alights right first, then "add ones" to the left of the smaller tensor or both(?), then BT over 1-dims ONLY
    - Matrix BT, aligns LEFT(?) first, BT over batch dims ONLY, and then check that matmul is valid with matching inner dims
        - For Matrice addition (elem-wise),  we can use tensor BT rules.
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
14. The most difficult lines:
    ```python
    self.register_buffer("tril", torch.ones([block_size,block_size])) # causal attention mask
    torch.matrix_fill(self.tril[:T, :T] == 0, float("-inf")) # mask out the future tokens
    qk_t = Q @ K.transpose(-2, -1) * self.head_size**-0.5 # MHA attention denominator
    self.heads = nn.ModuleList(MHA(head_size) for _ in range(n_heads)) # multiple heads in parallel
    x = torch.cat([h(x) for x in self.heads], dim=-1) # merge the parallel computed heads 
    self.blocks = nn.Sequential(*[MHABlock(n_embd, n_heads) for _ in range(n_layer)]) # GPT multiple blocks
    ```
# Advanced:
1. Vision Transformers
    - Why do we CONCAT the image embedding (CLS Embedding from ViT encoder) with the token embedding (from Decoder text backbone) 
        - Short Ans: Image is treated as separate token in the sequence. need to attend to it as a distinct element just like any other token
    - BUT add the positional embedding
        - Each token needs both content and position information in the same embedding. Addition merges them while keeping same dimensionality.
    - <image_pad> is a placeholder token for the image embedding. We eventually replace this with the actual image embedding (with same dimensionality as token embeddings)
        - BUT these image embeddings ARE NOT a distinct token in the text token vocabulary at all!
        - Marker tokens <vision_start> and <vision_end> are used to mark the start and end of the image embedding in the sequence and are in the token vocab
    - Number of <image_pad> tokens differs based on arch
        - Single CLS token, only accept single image -> Early CLIP-based models
        - Fixed multiple <image_pad> tokens, can accept single image. E.g. Llava-1.5 accept 336x336 image with each patch being 14x14 pixels
        - Dynamic Tokens depending on resolution -> # Low resolution (224×224): 256 tokens; # High resolution (448×448): 1024 tokens
        - Compressed Tokens (Learnable Reduction) -> BLIP-2 (Q-Former), compresses ViT output to fixed number like 32 compressed tokens always.
2. LoRa
    - [MSFT Implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
    - [TensorGym Easy Question](https://tensorgym.com/exercises/17)
