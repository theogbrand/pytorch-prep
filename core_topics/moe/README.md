Replace FFN in Transformer Block with Mixture of Experts (MoE) Block, everything else is the same.

"Sparsity" comes from how only the "Top-K" experts are used in computation at every token.