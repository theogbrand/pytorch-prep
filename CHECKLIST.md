# Fundamental
1. Debugging Transformers 
    - Forward Pass from scratch (Self/Cross-Attention, Positional/Token Embeddings, Batch/Layer Norm) **OK**
    - Backpropagation from scratch (Training), Chain Rules, Auto-Diff, Indexing Errors debugging **OK**
    - Tokenization (Implementing BPE) **OK**
    - Transformer Blocks *Optimizations* (RMSNorm, AdamW/RMSProp (LR scheduler > SGD), Attention Optimizations like FlashAttention, Sliding Window Attention, Long Context Length like RoPe/YaRN) **Now**
    - Debugging tensor shapes -> see NanoGPT/NanoChat/MicroGrad
2. KV Cache
    - Building Matrices **OK**
3. Basic distributed training implementations **Now, see NanoGPT/Chat**

Go through implementations of Quantization, LoRa, RMSNorm/LayerNorm, Sliding Window Attention, Long Context Length like RoPe/YaRN/ALiBi, FlashAttention, etc. 

Top-K/KNN, Speculative Decoding, etc. Ensure FP16/FP32 tensor array? Mixed Precision Training? Distributed Training?

Implement custom layer for... debug a transformer block... implement custom autograd for...

4. Basic Multimodal Transformers (VLM)
5. Top-K/KNN 
    - Common implementation pattern of "picking the K largest items" (BoN)
    - Look at how heaps data structures can help
6. Decoding Strategies / Fast Inference
    - Binary Search, Backtracking, Dijkstra
    - Speculative Decoding *See This*
7. Llama Architecture
8. (LSTMs) -> RNNs -> Transformers
9. Vision Language Models (VLMs)/[Image Transformer](https://arxiv.org/abs/1802.05751)/Vision Transformer (ViT) *NanoChat + SIGLIP*
10. Vision Language Action Models (VLAs)

# Advanced (if time permits)
11. Basic GPU Profiling *how to go fast*
12. Mixture of Experts *know conceptually*
13. Muon Optimizer (NanoChat), LoRa, RL (NanoChat) *Go Through these*

# Project and Knowledge Test
    - VL-PRM

References:
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [LucidRains Transformers](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py)
- [Stanford CS231N](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)
- [Stanford CS336](https://www.youtube.com/watch?v=6OBtO9niT00&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=7)
- [Umar Jamil ViT](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s)