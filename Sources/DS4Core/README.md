# DS4Core

The engine's **pure Swift, no-Metal** foundation. It contains model parsing,
tokenization, sampling, and the chat/tool prompt format. `DS4Metal` and
`DS4Engine` both build on this target, and this layer has the broadest unit-test
coverage under `Tests/DS4CoreTests`.

- **`Format/`**: on-disk formats and low-level helpers, including GGUF mmap,
  `Half` f16 conversion, and `KVCFile` disk checkpoints for KV cache state.
- **`Inference/`**: model shape, DeepSeek-V4 BPE tokenizer and control tokens,
  sampler, chat rendering, and DSML tool-call parsing.
- **`Streaming/`**: SSD cache planning and simulated memory locking used to
  reason about working sets before real runtime wiring.

There are no external dependencies and no link to Metal. That keeps this target
portable, fast to compile, and easy to test.
