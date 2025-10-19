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
    - torch.where(condition, x, y) for multiple checks; like a ternary operator but only for a single if-else
- Use output.copy_to(input) to copy the input tensor to the output tensor in-place for efficiency

## Convolution Operations
- 1D Conv (Stride, Dilation, Padding) -> see https://leetgpu.com/challenges/1d-convolution
    - we elem-wise multiply AND SUM for Convolution Operation, which is aka dot product (MatMul)
    - F.conv1d: needs (batch, channels, length)
    - F.conv2d: needs (batch, channels, height, width)
    - F.conv3d: needs (batch, channels, depth, height, width)

# Challenging (Medium to hard difficulty):
- Implement LoRA in a Linear Layer (TensorGym) - Hard
- Int8 Quantized MatMul in PyTorch (LeetGPU) - Hard