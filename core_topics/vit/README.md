Under the hood, nn.Linear uses nn.Parameter:

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias+ self.bias
```

Image -> Vision Encoder (Contains CLIP/SigLip to "project" image into a token sequence via 2D convolutions, then goes through a Transformer Encoder itself)-> Embedding of CLS token (different dims from text embeddings) -> Vision-Language Projection Module (MLP to project image embedding to text embedding space) -> Text Embedding Space (Now we can append CLS embedding between special image delimitter tokens <image_start> and <image_end> together with the text tokens) - CONCAT (usually at front) to sequence not add -> Any Decoder Text Backbone -> Output

# Vision Transformer (ViT)
- Essentially the "Image Encoder". 
    - Converts input image -> embedding of CLS token
        - image -> patches -> patch embeddings + positional embeddings -> transformer blocks -> embedding of CLS token

# Vision-Language Projection Module
- Project image embedding to language embedding space (text embedding of Decoder text backbone)
- Exactly like the final output projection layer in CLM, projects language embedding space to token vocabulary space
    - e.g. Image Embedding (3D) -> Projection Module -> Text Embedding Space (2D)
- Resulting Embedding then CONCAT with token embedding