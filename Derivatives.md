# 🧮 Common Derivatives Cheatsheet for AI Research Scientist Interviews

This sheet covers the derivatives you should **memorize cold** — especially for PyTorch autograd and manual gradient derivations.

---

## 🔹 Fundamental Derivatives

| Function | Derivative |
|-----------|-------------|
| d/dx (x^n) | n·x^(n−1) |
| d/dx (e^x) | e^x |
| d/dx (a^x) | a^x·ln(a) |
| d/dx (ln(x)) | 1/x |
| d/dx (log_a(x)) | 1 / (x·ln(a)) |

---

## 🔹 Trigonometric

| Function | Derivative |
|-----------|-------------|
| sin(x) | cos(x) |
| cos(x) | −sin(x) |
| tan(x) | sec²(x) |
| sec(x) | sec(x)·tan(x) |
| csc(x) | −csc(x)·cot(x) |
| cot(x) | −csc²(x) |

---

## 🔹 Inverse Trigonometric

| Function | Derivative |
|-----------|-------------|
| sin⁻¹(x) | 1 / √(1 − x²) |
| cos⁻¹(x) | −1 / √(1 − x²) |
| tan⁻¹(x) | 1 / (1 + x²) |

---

## 🔹 Exponential and Logarithmic Combinations

| Function | Derivative |
|-----------|-------------|
| e^(ax) | a·e^(ax) |
| ln(ax) | 1/x |
| x^x | x^x (1 + ln(x)) |

---

## 🔹 Neural Network Activations

| Function | Derivative |
|-----------|-------------|
| Sigmoid: σ(x) = 1 / (1 + e^(−x)) | σ(x)(1 − σ(x)) |
| tanh(x) | 1 − tanh²(x) |
| ReLU(x) | 1 if x>0 else 0 |
| LeakyReLU(x) | 1 if x>0 else α |
| Softplus(x) = ln(1 + e^x) | 1 / (1 + e^(−x)) = Sigmoid(x) |

---

## 🔹 PyTorch & Transformer-Relevant Gradients

| Function | Derivative |
|-----------|-------------|
| Softmax(xᵢ) = e^(xᵢ)/Σⱼ e^(xⱼ) | ∂yᵢ/∂xⱼ = yᵢ(δᵢⱼ − yⱼ) |
| Cross Entropy: −Σ y·log(ŷ) | ∂L/∂logits = ŷ − y |
| Mean(x) | 1/n |
| Variance(x) | 2(x − μ)/n |
| ||x||² = xᵀx | 2x |
| BatchNorm μ = mean(x), σ² = var(x) | ∂y/∂x includes normalization and affine params |

---

## 🔹 Core Calculus Rules

| Rule | Formula |
|------|----------|
| Chain rule | (f(g(x)))′ = f′(g(x))·g′(x) |
| Product rule | (uv)′ = u′v + uv′ |
| Quotient rule | (u/v)′ = (u′v − uv′) / v² |

---

✅ **Tip for PyTorch interviews**: be able to derive these manually and verify with `torch.autograd.gradcheck`.