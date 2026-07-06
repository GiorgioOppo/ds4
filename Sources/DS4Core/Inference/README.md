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
- **`LoadProgress.swift`** is a thread-safe singleton progress reporter: the
  model-load path writes milestones and per-unit advances, and the UI polls
  `snapshot` to render a determinate bar.

The user-facing sampling knobs (temperature, repetition penalty, the fixed
top-k) are documented in the root
[Configuration Reference](../../../README.md#configuration-reference).
