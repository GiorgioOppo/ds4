**English** | [Italiano](README.it.md)

# Laguna S 2.1 tokenizer

`LagunaTokenizer` implements GPT-2 byte-level BPE with the Laguna
pre-tokenizer: runs of LF bytes are separated from non-newline spans first,
then each span goes through the GLM4-style split shape with single-digit
number groups.  The newline pre-split is observable for CRLF — CR stays in the
preceding span and LF starts a new one, so they never merge into one BPE
piece.

The model initializer validates `general.architecture`, the tokenizer tables,
and the required control tokens (`<assistant>`, `</assistant>`, `<think>`,
`</think>`, `<tool_call>`, `</tool_call>`); BOS/EOS ids are required metadata
and the end-of-turn id falls back to the `</assistant>` vocabulary entry.
Upstream selects the pre-tokenizer by model family, not from
`tokenizer.ggml.pre`, so that key is deliberately not gated here.  A model-free
internal initializer and the pure `LagunaPretokenizer` splitter support
deterministic unit tests without downloading or mapping a Laguna GGUF.

This layer does not select an inference backend. Model detection may recognize
Laguna while the Metal graph remains explicitly unavailable.
