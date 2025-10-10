"""
Word2Vec Toy Example: Skip-gram and CBOW
==========================================

This implementation demonstrates the mathematical fundamentals of Word2Vec
with a tiny vocabulary and small dimensions for educational purposes.

Setup:
- Vocabulary: ["the", "cat", "sat", "on", "mat"] (5 words)
- Embedding dimension: 3
- Example sentence: "the cat sat on the mat"
- Window size: 2
"""

import numpy as np
from typing import List, Tuple


class Word2VecToy:
    """
    Toy implementation of Word2Vec with both Skip-gram and CBOW architectures.
    """

    def __init__(self, vocabulary: List[str], embedding_dim: int = 3):
        """
        Initialize Word2Vec model.

        Args:
            vocabulary: List of words in the vocabulary
            embedding_dim: Dimension of word embeddings
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.embedding_dim = embedding_dim

        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Initialize weight matrices with small random values
        # W: Input to hidden layer (vocab_size x embedding_dim)
        # This is the embedding matrix - each row is a word's embedding
        self.W = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1

        # W_prime: Hidden to output layer (embedding_dim x vocab_size)
        # This is the context matrix
        self.W_prime = np.random.randn(self.embedding_dim, self.vocab_size) * 0.1

        print(f"Initialized Word2Vec with vocabulary size {self.vocab_size} "
              f"and embedding dimension {self.embedding_dim}")
        print(f"W shape: {self.W.shape}, W' shape: {self.W_prime.shape}")

    def one_hot_encode(self, word_idx: int) -> np.ndarray:
        """
        Create one-hot encoding for a word.

        Args:
            word_idx: Index of the word in vocabulary

        Returns:
            One-hot encoded vector of shape (vocab_size,)
        """
        one_hot = np.zeros(self.vocab_size)
        one_hot[word_idx] = 1
        return one_hot

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities.

        Args:
            scores: Raw output scores

        Returns:
            Probability distribution
        """
        # Subtract max for numerical stability
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def forward_skipgram(self, center_word: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for Skip-gram: predict context words from center word.

        Args:
            center_word: The center word
            verbose: Whether to print detailed steps

        Returns:
            Tuple of (hidden_layer, output_probabilities)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"SKIP-GRAM: Predicting context words from center word '{center_word}'")
            print(f"{'='*70}")

        # Step 1: One-hot encoding
        center_idx = self.word_to_idx[center_word]
        one_hot_input = self.one_hot_encode(center_idx)

        if verbose:
            print(f"\nStep 1: One-hot encoding of '{center_word}'")
            print(f"  Index: {center_idx}")
            print(f"  One-hot: {one_hot_input}")

        # Step 2: Input to hidden layer
        # Since input is one-hot, this simply picks the corresponding row from W
        hidden_layer = self.W[center_idx]  # Equivalently: one_hot_input @ self.W

        if verbose:
            print("\nStep 2: Input to hidden layer (embedding lookup)")
            print(f"  W shape: {self.W.shape}")
            print(f"  Selected row {center_idx} from W:")
            print(f"  Hidden layer h = {hidden_layer}")

        # Step 3: Hidden to output layer
        output_scores = hidden_layer @ self.W_prime

        if verbose:
            print("\nStep 3: Hidden to output layer")
            print(f"  W' shape: {self.W_prime.shape}")
            print("  Output scores = h x W'")
            print(f"  Raw scores: {output_scores}")
            print("\n  Detailed calculation for each word:")
            for word_idx, word in enumerate(self.vocabulary):
                score = np.dot(hidden_layer, self.W_prime[:, word_idx])
                terms = ' + '.join([f'{hidden_layer[i]:.2f}x{self.W_prime[i, word_idx]:.2f}'
                                  for i in range(self.embedding_dim)])
                print(f"    {word}: {terms} = {score:.4f}")

        # Step 4: Softmax to get probabilities
        probabilities = self.softmax(output_scores)

        if verbose:
            print("\nStep 4: Softmax probabilities")
            for word_idx, word in enumerate(self.vocabulary):
                print(f"  P({word} | {center_word}) = {probabilities[word_idx]:.4f}")

        return hidden_layer, probabilities

    def forward_cbow(self, context_words: List[str], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for CBOW: predict center word from context words.

        Args:
            context_words: List of context words
            verbose: Whether to print detailed steps

        Returns:
            Tuple of (hidden_layer, output_probabilities)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"CBOW: Predicting center word from context words {context_words}")
            print(f"{'='*70}")

        # Step 1: One-hot encoding for all context words
        context_indices = [self.word_to_idx[word] for word in context_words]

        if verbose:
            print("\nStep 1: One-hot encoding of context words")
            for word, idx in zip(context_words, context_indices):
                one_hot = self.one_hot_encode(idx)
                print(f"  '{word}' (index {idx}): {one_hot}")

        # Step 2: Look up embeddings and average them
        context_embeddings = [self.W[idx] for idx in context_indices]
        hidden_layer = np.mean(context_embeddings, axis=0)

        if verbose:
            print("\nStep 2: Look up embeddings and average")
            for word, embedding in zip(context_words, context_embeddings):
                print(f"  '{word}' embedding: {embedding}")
            print(f"  Average hidden layer h = {hidden_layer}")
            embeddings_str = ' + '.join([str(e) for e in context_embeddings])
            print(f"  Calculation: ({embeddings_str}) / {len(context_embeddings)}")

        # Step 3: Hidden to output layer
        output_scores = hidden_layer @ self.W_prime

        if verbose:
            print("\nStep 3: Hidden to output layer")
            print("  Output scores = h x W'")
            print(f"  Raw scores: {output_scores}")
            print("\n  Detailed calculation for each word:")
            for word_idx, word in enumerate(self.vocabulary):
                score = np.dot(hidden_layer, self.W_prime[:, word_idx])
                terms = ' + '.join([f'{hidden_layer[i]:.2f}x{self.W_prime[i, word_idx]:.2f}'
                                  for i in range(self.embedding_dim)])
                print(f"    {word}: {terms} = {score:.4f}")

        # Step 4: Softmax to get probabilities
        probabilities = self.softmax(output_scores)

        if verbose:
            print("\nStep 4: Softmax probabilities")
            for word_idx, word in enumerate(self.vocabulary):
                print(f"  P({word} | {context_words}) = {probabilities[word_idx]:.4f}")

        return hidden_layer, probabilities

    def cross_entropy_loss(self, probabilities: np.ndarray, target_word: str) -> float:
        """
        Compute cross-entropy loss.

        Args:
            probabilities: Predicted probabilities
            target_word: Target word

        Returns:
            Cross-entropy loss value
        """
        target_idx = self.word_to_idx[target_word]
        # Cross-entropy loss: -log(p(target))
        loss = -np.log(probabilities[target_idx] + 1e-10)  # Add small value to avoid log(0)
        return loss

    def demonstrate_skipgram(self, center_word: str, context_words: List[str]):
        """
        Demonstrate Skip-gram training on a single example.

        Args:
            center_word: Center word
            context_words: List of context words
        """
        print(f"\n{'#'*70}")
        print("# SKIP-GRAM DEMONSTRATION")
        print(f"# Center word: '{center_word}' -> Context words: {context_words}")
        print(f"{'#'*70}")

        # Skip-gram generates separate training examples for each context word
        total_loss = 0
        for context_word in context_words:
            print(f"\n{'-'*70}")
            print(f"Training example: '{center_word}' -> '{context_word}'")
            print(f"{'-'*70}")

            _hidden, probs = self.forward_skipgram(center_word, verbose=True)
            loss = self.cross_entropy_loss(probs, context_word)
            total_loss += loss

            print(f"\nLoss for predicting '{context_word}': {loss:.4f}")

        avg_loss = total_loss / len(context_words)
        print(f"\n{'-'*70}")
        print(f"Average loss for all context words: {avg_loss:.4f}")
        print('-'*70)

    def demonstrate_cbow(self, context_words: List[str], center_word: str):
        """
        Demonstrate CBOW training on a single example.

        Args:
            context_words: List of context words
            center_word: Center word to predict
        """
        print(f"\n{'#'*70}")
        print("# CBOW DEMONSTRATION")
        print(f"# Context words: {context_words} -> Center word: '{center_word}'")
        print(f"{'#'*70}")

        _hidden, probs = self.forward_cbow(context_words, verbose=True)
        loss = self.cross_entropy_loss(probs, center_word)

        print(f"\nLoss for predicting '{center_word}': {loss:.4f}")


def main():
    """
    Main demonstration following the exact example from the explanation.
    """
    print("="*70)
    print("WORD2VEC TOY EXAMPLE: Understanding the Mathematics")
    print("="*70)

    # Setup
    vocabulary = ["the", "cat", "sat", "on", "mat"]
    sentence = "the cat sat on the mat"
    window_size = 2

    print("\nSetup:")
    print(f"  Vocabulary: {vocabulary}")
    print(f"  Sentence: '{sentence}'")
    print(f"  Window size: {window_size}")
    print("  Embedding dimension: 3")

    # Initialize model
    model = Word2VecToy(vocabulary, embedding_dim=3)

    # Set specific weight matrices to match the example
    print(f"\n{'-'*70}")
    print("Setting weight matrices to match the example...")
    print('-'*70)

    model.W = np.array([
        [0.1, 0.2, 0.3],   # the
        [0.4, 0.5, 0.6],   # cat
        [0.7, 0.8, 0.9],   # sat
        [1.0, 1.1, 1.2],   # on
        [1.3, 1.4, 1.5]    # mat
    ])

    model.W_prime = np.array([
        [-0.1,  0.2, -0.3,  0.4, -0.5],
        [ 0.6, -0.7,  0.8, -0.9,  1.0],
        [-1.1,  1.2, -1.3,  1.4, -1.5]
    ])

    print("\nW (Embedding Matrix):")
    for idx, word in enumerate(vocabulary):
        print(f"  {word:>5s}: {model.W[idx]}")

    print("\nW' (Context Matrix):")
    print(model.W_prime)

    # Example 1: Skip-gram
    # Center word: "sat", Context words: ["cat", "on"]
    center_word = "sat"
    context_words = ["cat", "on"]
    model.demonstrate_skipgram(center_word, context_words)

    # Example 2: CBOW
    # Context words: ["cat", "on"], Center word: "sat"
    model.demonstrate_cbow(context_words, center_word)

    # Key differences summary
    print(f"\n{'#'*70}")
    print("# KEY DIFFERENCES BETWEEN SKIP-GRAM AND CBOW")
    print(f"{'#'*70}")
    print("""
Skip-gram:
  - Input: Center word (one word)
  - Output: Context words (multiple words)
  - Training: Each (center, context) pair is a separate training example
  - In our example: "sat" -> "cat" and "sat" -> "on" are TWO separate examples
  - Advantage: Treats each context relationship individually
  - Disadvantage: Slower training (more examples)

CBOW (Continuous Bag of Words):
  - Input: Context words (multiple words, averaged)
  - Output: Center word (one word)
  - Training: All context words together predict the center word
  - In our example: ["cat", "on"] -> "sat" is ONE example
  - Advantage: Faster training (fewer examples)
  - Disadvantage: Smooths over individual context relationships

Mathematical difference:
  - Skip-gram: h = W[center_word_idx]
  - CBOW: h = mean(W[context_word_1_idx], W[context_word_2_idx], ...)
    """)


if __name__ == "__main__":
    main()
