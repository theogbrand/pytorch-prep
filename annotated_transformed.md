A Transformer takes as input a word, passes it through a stack of encoder blocks (6 or more sequentially). The final encoder block outputs to a stack of decoder blocks of the same number. The final decoder block outputs a word. 
<img src="images/2025-10-07-13-15-28.png" alt="stacked encoder-decoder blocks" width="600">

every word -> single embedding through embd algorithm, each word is embedded into a vector of size 512 (or the length of the longest sentence in training data)

Encoders identical in containing self-attention layer and FFNN layer. Point of self-attention 


