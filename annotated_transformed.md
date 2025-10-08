A Transformer takes as input a word, passes it through a stack of encoder blocks (6 or more sequentially). The final encoder block outputs to a stack of decoder blocks of the same number. The final decoder block outputs a word. 
<img src="images/2025-10-07-13-15-28.png" alt="stacked encoder-decoder blocks" width="600">

every word -> single embedding through embd algorithm, each word is embedded into a vector of size 512 (or the length of the longest sentence in training data)
Word Embedding Algorithm used to convert words (text) from a Vocabulary to a corresponding vector of real numbers.
- instead of Bag Of Words (one-hot encoding) with high dimensionality, sparse vectors, ideally our WEA should have two advantageous properties:
    - Dimensionality reduction (efficient representation)
    - Contextual similarity (for semantic parsing - NLU) e.g. (King - Man + Woman = Queen, Word2Vec; Paris - France + Italy = Rome)
- Word2Vec implemented as Continuous Bag Of Words (CBOW) and Skip-Gram
    - 2-layer neural network, input_sz = output_sz = vocab_size (order of 10K, 100K); hidden_sz = embedding_sz (order of 300, 200, 100)
    - Hidden layer actually becomes the word embedding vector we use, and the num_neurons/features is a Hyperparameter we determine 
- Trained using hierarchical softmax (or negative sampling) to learn dense, low-dimensional word embeddings from the high dimensional, sparse one-hot encoded input vocab

Encoders identical in containing self-attention layer and FFNN layer. Point of self-attention 


