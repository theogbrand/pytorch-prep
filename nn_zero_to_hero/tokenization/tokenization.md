- UTF-8 Encoding is the most "efficient"/least wasteful amongst all, and backwards compatibilities with ASCII
    - *Unicode* (vocab of 1M diff characters from letters, emojis, symbols from all languages) - ord(x) converts to unicode; then x.encode('utf-8') converts to bytes
        - Downsides of Unicode is only defines 150K vocab, constantly changing standard
    - BUT **Unicode** has encodings which are stable enough to be used, taking Unicode code points and converting to bytes strings (1 to 4 bytes) - variable length encoding
    - Byte usage: Depending on the character's Unicode code point, UTF-8 uses 1 to 4 bytes:
        - Characters 0-127 (basic ASCII like letters, digits, punctuation) use 1 byte
        - Characters 128-2,047 use 2 bytes
        - Characters 2,048-65,535 use 3 bytes -> "안" unicode code point is 50504
        - Characters 65,536 and beyond use 4 bytes
    - Code point is what the character is. Bytes are how it's physically stored.
        - range 0 to 1,114,111; *always 1 per character*, use ord(x) to get the code point.
    - Byte Values are always 0 to 255 (8 positions to store the value); each byte is 8 bits, so 2^8 = 256 possible values.
        - use x.encode('utf-8') to get the bytes; could be **1, 2, 3, or 4 bytes per character** 
- list(x.encode('utf-8')) converts to list of bytes, e.g. list("안".encode('utf-8')) -> [236, 149, 136]
    - match the encodings to the UTF-8 specification (deterministic):
        **UTF-8 byte structure:**
        The key is the leading bits of each byte tell you what's happening:
        - 0xxxxxxx = A complete 1-byte character
        - 110xxxxx = Start of a 2-byte character (followed by 1 continuation byte)
        - 1110xxxx = Start of a 3-byte character (followed by 2 continuation bytes)
        - 11110xxx = Start of a 4-byte character (followed by 3 continuation bytes)
        - 10xxxxxx = A continuation byte (always starts with 10)
        - Example: "안" -> [236, 149, 136]
            - 236: 1110xxxx (start of 3-byte character, which means to expect 2 more continuation bytes)
            - 149: 10xxxxxx (continuation byte)
            - 136: 10xxxxxx (continuation byte)

            **Step 1: Break down the bytes**
            ```
            236 → 11101100
            149 → 10010101
            136 → 10001000
            ```

            **Step 2: Extract the meaningful bits**

            UTF-8's 3-byte structure is: `1110xxxx 10xxxxxx 10xxxxxx`

            The `x`s are the actual information bits. Extracting them:
            ```
            11101100  →  1110[1100]  →  extract: 1100
            10010101  →  10[010101]  →  extract: 010101
            10001000  →  10[001000]  →  extract: 001000

            Combined bits: 1100 010101 001000
            ```

            **Step 3: Convert back to decimal**
            ```
            1100010101001000 (binary) = 50504 (decimal)

            1 to 4 byte patterns:
            ```text
            'A'      # U+0041 = 65
            encode:  [65]
            binary:  01000001           (1-byte pattern: 0xxxxxxx) 

            'é'      # U+00E9 = 233
            encode:  [195, 169]
            binary:  11000011 10101001  (2-byte pattern: 110xxxxx 10xxxxxx) 

            '中'     # U+4E2D = 20013
            encode:  [228, 184, 173]
            binary:  11100100 10111000 10101101  (3-byte pattern) 

            '😀'     # U+1F600 = 128512
            encode:  [240, 159, 152, 128]
            binary:  11110000 10111111 10011000 10000000  (4-byte pattern) 
            ```
- UTF-8 Encoding Scheme for LLM
    - Naive raw byte-level UTF encoding (only 256 possible token vocab), results in sequences too long
    - Want something that support tunable vocab size, but with UTF-8 encoding of strings (cos of relative stability)
    - Thinking: encoding of text -> some representation, long enough to capture semantics, 
- Point is we have finite context length to attend to in Transformer, how do we pick the right tokens 

# Byte Pair Encoding (BPE) Algorithm
1. Get Byte-Level encoding using UTF-8 encoding -> returns list of bytes, each 0-255. 
2. Analyze the frequency of pairs of bytes in the data, find the pair that appears most frequently, and mint a new token to "replace" this pair of bytes.
3. Repeat the process until the desired vocab size is reached.
    - the newly minted tokens can also be replaced if they still contain the most frequent pair

Example:
Seq=11 to 9 tokens: aaabdaaabac -> XabdXabac
Vocab=4 to 5: abcdx

Seq=9 to 7 tokens: aaabdaaabac -> XYdXYac
Vocab=5 to 6: ABCDXY

Seq=7 to 5 tokens: XYdXYac -> ZdZac
Vocab=6 to 7: ABCDXYZ

Overall: Sequence compressed from 11 to 5 tokens, vocab size increased from 4 to 7.

In General, Tokenizer which tokenizes text into much smaller sequence length is advantageous for the model as it can attend to more context tokens to learn better semantic representations
    - Examples (negative ones): 
        - space tokens existing as standalone tokens without multiple space tokens having their own tokens (for code gen especially)
        - Korean characters of common phrases like (Hello -> AnnYeongHaSeYo) not existing as a single token
        - JSON contains more characters than YAML for the same data representation

### Training Tokenizer is completely independent from training the model but also coupled in a sense that the same tokenizer should be used for training data and for using during inference

### BUT we can use a completely different subset of the training data to train the tokenizer, for example when we want more multilingual token support, we use a subset of training data (or could be completely different dataset) with more multilingual tokens, resulting in more multilingual merges.

SolidGoldMagikarp is a consequence of the phenomenon. In Tokenization training set, used lots of reddit data containing this token to train GPT2 tokenizer, resulting in a single "SolidGoldMagikarp" token. BUT during training, the training data did not contain this token, so the embedding for this token is never trained, resulting in a "random" embedding for this token. During inference, the model uses the random embedding to create output sequences, resulting in Gibberish. (like "unallocated memory" for SolidGoldMagikarp token)

# Naive BPE Optimizations

## Dealing with common words with spurious variations (e.g. punctionation, capitalization, etc. dog. dog? dog!)
- Enforce how some characters should never be merged together for sure, using **REGEX pattern** to match and "chunk" text first into a list of strings, then tokenize each chunk individually before merging them together.
    - Issues with regex pattern like not case sensitive
- The REGEX pattern first splits your text into individual words in a list following the regex rules, THEN tokenizes individually independently before merging them together.