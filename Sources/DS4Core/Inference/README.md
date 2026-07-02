# DS4Core/Inference

Pure inference-facing pieces that do not touch the GPU.

- **`Tokenizer.swift`** implements BPE with DeepSeek-V4 control tokens such as
  BOS/EOS, `<｜User｜>`, `<think>`, and `｜DSML｜`. Important entry points include
  `tokenizeRenderedChat` and `tokenText`.
- **`ChatTools.swift`** defines `ToolSpec`, `ToolCall`, and `ChatTurn`; renders
  chat/tool prompts in the DSML format; and provides `ToolCallParser`, which
  extracts tool calls from generated text.
- **`Sampler.swift`** implements temperature, top-k/top-p, min-p, and repetition
  penalty sampling.
- **`ModelShape.swift`** describes the model dimensions consumed by the runtime.
