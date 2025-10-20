# Derivative Rules Cheat Sheet

## 1. Power Rule
**f(x) = x^n → f'(x) = n·x^(n-1)**
- Example: x³ → 3x²

## 2. Constant Multiple Rule
**f(x) = c·g(x) → f'(x) = c·g'(x)**
- Example: 5x² → 10x

## 3. Sum/Difference Rule
**f(x) = g(x) ± h(x) → f'(x) = g'(x) ± h'(x)**
- Example: x³ + 2x² → 3x² + 4x

## 4. Product Rule ⭐
**f(x) = g(x)·h(x) → f'(x) = g'(x)·h(x) + g(x)·h'(x)**
- Example: x²·x³ → 2x·x³ + x²·3x² = 5x⁴

## 5. Quotient Rule
**f(x) = g(x)/h(x) → f'(x) = [g'(x)·h(x) - g(x)·h'(x)] / [h(x)]²**
- Example: x³/x² → (3x²·x² - x³·2x)/x⁴ = 1

## 6. Chain Rule ⭐⭐
**f(x) = g(h(x)) → f'(x) = g'(h(x))·h'(x)**
- Example: (2x + 1)³ → 3(2x + 1)²·2 = 6(2x + 1)²
- **Critical for backpropagation in neural networks**

---

## Common Interview Questions

### Basic
- Calculate derivative of 3x⁴ + 2x² - 5
- Find derivative of x⁵/x²

### Intermediate
- Derivative of (x² + 3)(2x - 1) [Product Rule]
- Differentiate (3x² + 2x)⁵ [Chain Rule]

### Advanced
- Derivative of (x² + 1)³/(2x - 5) [Chain + Quotient]
- Explain chain rule's role in backpropagation
- Apply chain rule through 3-layer composite function

### ML Context Questions
- Why is chain rule essential for neural network training?
- How does gradient descent use derivatives?
- Explain vanishing/exploding gradients in context of chain rule