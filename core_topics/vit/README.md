Under the hood, nn.Linear uses nn.Parameter:

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias+ self.bias
```
