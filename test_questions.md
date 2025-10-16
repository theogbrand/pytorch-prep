# Normalization and Residual Connections:
- Backward pass derivation for BatchNorm (∂L/∂x)? (step-by-step BatchNorm steps, then show by-hand chain rule)
- "You're training a 24-layer Transformer. Gradients vanish in early layers. What's wrong?"
- "Your model outputs grow unbounded. Which component might be broken?"
- If residuals are so good, why not add one after every single layer?"
- "What's the effective depth of a 12-layer Transformer with residuals?" (Hint: Multiple paths!)
- "Show mathematically why gradients 'skip' through residual connections"
- "Can you have a 'learned skip' where the network decides when to skip?" (Yes! MoE, etc.)