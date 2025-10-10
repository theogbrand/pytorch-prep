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
- Once trained, the Word Embedding of a word is an O(1) operation on the embedding matrix
- to revisit math

Encoders identical in containing self-attention layer and FFNN layer. Point of self-attention 

# Self-Attention Layer
1. After going through the embedding algorithm, the vectors approach the query, key, and value weight matrices that are created during training. 
 - This is where we take the *dot product of each word vector with the q, k, and v* weight matrices (Wq, Wk, Wv), Returning the vector projection of each word embedding into the query, key, and value spaces. (qi, ki, vi, where i is the word index corresponding to its embedding - every word gets its own query, key, and value vectors)

2. *Scaled Dot-Product Attention* "Score"
    - This determines how much attention to pay in the sentence with respect to this word itself and the other words in the sentence. (the "self" in self-attention)
    - Calculated by dot product of query and key vectors (qk), and scaled by dk**0.5 (for numerical stability)
    - Every word has a **score** with respect to every other word in the sentence. (Hence *n-squared scores for a sentence of length n, and the quadratic problem*)
    - Note the difference in size of the embedding vector (usually 512) and the q/k/v vectors (usually 64)
    <img src="images/2025-10-10-15-13-12.png" alt="query-key-similarity" width="700">
    - The result is a tensor of shape (batch, num_heads, seq_len_q, seq_len_k)
    - Queries represent "what context the word in question needs" and Keys represent "what context the word in question provides"
    - Hence we want a distribution of these "possible context words" summing to 1, and the softmax function, with dim=-1 enables this, normalizing all possible keys for each query (recall attention scores are calculated with respect to the word itself and the other words in the sentence)
