# Normalization and Residual Connections:
- Backward pass derivation for BatchNorm (∂L/∂x)? (step-by-step BatchNorm steps, then show by-hand chain rule)
- "You're training a 24-layer Transformer. Gradients vanish in early layers. What's wrong?"
- "Your model outputs grow unbounded. Which component might be broken?"
- If residuals are so good, why not add one after every single layer?"
- "What's the effective depth of a 12-layer Transformer with residuals?" (Hint: Multiple paths!)
- "Show mathematically why gradients 'skip' through residual connections"
- "Can you have a 'learned skip' where the network decides when to skip?" (Yes! MoE, etc.)

# Backprop
- Do you know how all operations (+-*/), sum/mean/max/exp work in backprop?
- Whenever c = a*b (like hpreact = bngain * bnraw + bnbias), local derivative is just dc/da = b; likewise dbngain = bnraw * dhpreact
- whenever there is y=mx+c the gradient of c just "flows through" as in dc = (1 * dy)
- What is **Bessel's correction** when calculating variance? (1/m is the biased estimate, 1/(m-1) is the unbiased estimate) why is 1/m used in inference but 1/(m-1) used in training?
    - minibatches are a small sample of entire population, so using unbiased estimate (1/(m-1)) is more accurate in training set
    - Kaparthy just says use Bessel's Corretion, unbiased=True for both training/inference for consistency
- How are gradients passed specifically for the operations tensor.mean(), tensor.max(), tensor.sum() where the operation involves "squashing" of the terms in the tensor