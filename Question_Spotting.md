# Tokenization
- Write BPE algorithm from scratch in Python
- Encode and Decode Functions, given init Vocab and merges
- Some BPE v.s. SentencePiece 

# BackProp:
- Backprop by hand Cross-Entropy Loss for a simple 2-layer MLP with 1 hidden layer
    - BatchNorm backward pass by hand

# PyTorch:
- Implement Softmax, MLP Forward Pass, Attention Block, Batch/Layer Norm, Custom ReLU using torch.nn.Functional, then without
    - Instead of loss.backward(), manually implement backprop for particular operation and show it adds up.
- Botched implementation of above, fix it.
- Custom ReLU mask: 
    - mask = (input > 0).float() for single check
        - Returns 1D tensor of True/False values, then cast to float 0.0/1.0. Since row vector can only sum across dims=0 or dims=-1
    - torch.where(condition, x, y) for multiple checks; like a ternary operator but only for a single if-else
- Use output.copy_(input) to copy the input tensor to the output tensor in-place for efficiency
- for 2D array, instead of # output[:] = (input==K).sum(dim=1).sum(dim=0) use torch.sum(input==K, dim=(0,1), out=output)
- torch.mul(input, torch.sigmoid(input), out=output) is equivalent to input * torch.sigmoid(input) - elem-wise operation since both shapes match

- Common Ops:
    - torch.mul(input, torch.sigmoid(input), out=output)
    - torch.nn.functional.softmax(input, dim=1)
    - torch.nn.functional.relu/sigmoid/tanh(input)
    - torch.nn.modules.activations.SiLU()

Notes:
## Use Modules when:
- Building models with nn.Sequential
- Want consistent, clean model definitions
- Following PyTorch conventions for model building

Use Functional when:
- Writing custom forward passes with complex logic
- One-off operations in notebooks
- You prefer functional programming style

## Convolution Operations
- 1D Conv (Stride, Dilation, Padding) -> see https://leetgpu.com/challenges/1d-convolution
    - we elem-wise multiply AND SUM for Convolution Operation, which is aka dot product (MatMul)
    - F.conv1d: needs (batch, channels, length)
    - F.conv2d: needs (batch, channels, height, width)
    - F.conv3d: needs (batch, channels, depth, height, width)

# Challenging (Medium to hard difficulty):
- Implement LoRA in a Linear Layer (TensorGym) - Hard
- Int8 Quantized MatMul in PyTorch (LeetGPU) - Hard