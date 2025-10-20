- The "level of abstraction" of expressions defined in a Forward/Backward pass is flexible. As long as define what is the local derivative of the operation, the Backward Pass will work.
- For elem wise mul (torch.mul), the sequence of tensors don't matter, but for matmul YES IT DOES! (which is why for linear transformations the Backward Pass formula is ALWAYS the same)
    - for FP: x1 @ w1 + c; BP: dLoss/dw1 = x1.T @ dLoss/dout; dLoss/dc = dLoss/dout.sum(dim=0); dLoss/dx1 = dLoss/dout @ w1.T
- Local derivative of y = torch.mean(x) -> 1/N * dy/dx
    - 1/N because torch.mean() returns a single scalar value, but we need to pass the gradient to each element of x
- logits = f_t @ w_t + b_t
    - dw_t = f_t.T @ dlogits
    - db_t = dlogits.sum(dim=0)
- when dw_t and db_t is calculated, we use w_t -= learning_rate * dw_t and b_t -= learning_rate * db_t; and NOT "+=" because gradient points towards increasing the loss, but we want to decrease the loss
    - for p in n.parameters(): p.data += -0.1 * p.grad
    - Gradient vector gives us direction to increase the loss. We want to decrease loss, so must add the negative sign when updating parameters with gradients
- BP Gradients:
    - dLoss/dLoss = 1 (base case) -> **We ALWAYS have out.grad**
    - For "addition" operation, local derivative is always 1. 
    - For "multiplication" operation, local derivative is the other operand
    - For "division" operation (y = a/b), local derivative is: dy/da = 1/b, dy/db = -a/b²
    - For "power" operation, local derivative is the power * the operand raised to the power - 1
    - For "square" operation, local derivative is 2 * the operand
    - For "square root" operation, local derivative is 0.5 * the operand raised to the power - 0.5
    - For "exponential" operation, local derivative is the exponential of the operand
    - For "logarithm" operation, local derivative is 1 / the operand
    - For "sine" operation, local derivative is the cosine of the operand
    - For "cosine" operation, local derivative is the negative sine of the operand
    - For "tangent" operation, local derivative is the secant squared of the operand
- Single line MSE:  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
- Attention "Math" trick: 
    - torch.tril(torch.ones(3, 3)) -> 
    ```python
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    ```
    MatMul this with attention scores (Q @ K^T) to get a mask that is 0 for future tokens and 1 for present tokens.