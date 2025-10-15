# Fundamental
1. Debugging Transformers 
    - Forward Pass from scratch (Self/Cross-Attention, Positional/Token Embeddings, Batch/Layer Norm) **OK**
    - Backpropagation from scratch (Training), Chain Rules, Auto-Diff, Indexing Errors debugging **Now**
    - Tokenization (Implementing BPE) **Later**
    - Transformer Blocks Extensions (RMSNorm, AdamW/RMSProp (LR scheduler > SGD), Attention Optimizations like FlashAttention, Sliding Window Attention, Long Context Length like RoPe/YaRN) **Later**
    - Debugging tensor shapes -> see NanoGPT/NanoChat/MicroGrad
2. KV Cache
    - Building Matrices **OK**
3. Basic distributed training implementations
4. Top-K/KNN 
    - Common implementation pattern of "picking the K largest items" (BoN)
    - Look at how heaps data structures can help
5. Decoding Strategies
    - Binary Search, Backtracking, Dijkstra
    - Speculative Decoding
6. Llama Architecture
7. (LSTMs) -> RNNs -> Transformers
8. Vision Language Models (VLMs)/[Image Transformer](https://arxiv.org/abs/1802.05751)/Vision Transformer (ViT)
9. Vision Language Action Models (VLAs)

# Advanced (if time permits)
10. Basic GPU Profiling
11. Mixture of Experts
12. Muon Optimizer, LoRa, RL

# Project and Knowledge Test
    - VL-PRM

References:
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [LucidRains Transformers](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py)
- [Stanford CS 336](https://www.youtube.com/watch?v=6OBtO9niT00&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=7)
- [Umar Jamil ViT](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s)