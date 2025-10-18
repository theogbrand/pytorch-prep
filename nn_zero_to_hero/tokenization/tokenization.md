- UTF-8 Encoding is the most "efficient" amongst all, but downside is 256 characters only. 
    - *Unicode* (vocab of 1M diff characters from letters, emojis, symbols from all languages) - ord(x) converts to unicode; then x.encode('utf-8') converts to bytes
        - Downsides of Unicode is only defines 150K vocab, constantly changing standard
    - BUT **Unicode** has encodings which are stable enough to be used, taking Unicode code points and converting to bytes strings (1 to 4 bytes) - variable length encoding
    - Byte usage: Depending on the character's Unicode code point, UTF-8 uses 1 to 4 bytes:
        - Characters 0-127 (basic ASCII like letters, digits, punctuation) use 1 byte
        - Characters 128-2,047 use 2 bytes
        - Characters 2,048-65,535 use 3 bytes -> "안" unicode code point is 50504
        - Characters 65,536 and beyond use 4 bytes
    - Code point is what the character is. Bytes are how it's physically stored.
        - range 0 to 1,114,111; always 1 per character, use ord(x) to get the code point.
    - Byte Values are always 0 to 255 (8 positions to store the value); each byte is 8 bits, so 2^8 = 256 possible values.
        - use x.encode('utf-8') to get the bytes; could be 1, 2, 3, or 4 bytes per character for utf-8 encoding specifically
- x.encode('utf-8') converts to list of bytes, e.g. "안".encode('utf-8') -> [236, 149, 136]
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
- UTF-8 as Encoding Scheme for LLMs
    - When used naively, all our text will be "stretched out" into long sequences of bytes, hard for model to attend to enough context to capture semantics
    - Thinking: encoding of text -> some representation, long enough to capture semantics, 
- Point is we have finite context length to attend to in Transformer, how do we pick the right tokens 