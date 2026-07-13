# DS4Core inference-facing components

Pure inference-facing pieces that do not touch the GPU.

- **`Tokenization/Tokenizer.swift`** implements BPE with DeepSeek-V4 control tokens such as
  BOS/EOS, `<｜User｜>`, `<think>`, and `｜DSML｜`. Important entry points include
  `tokenizeRenderedChat` and `tokenText`.
- **`Conversation/Models/ConversationModels.swift`** defines `ToolSpec`,
  `ToolCall`, and `ChatTurn`; `Conversation/DSML/ChatRenderer.swift` renders
  chat/tool prompts; and `ToolCallParser.swift` extracts calls from generated
  DSML.
- **`Generation/Sampler.swift`** implements temperature, top-k/top-p, min-p, and repetition
  penalty sampling.
- **`Model/ModelShape.swift`** describes the model dimensions consumed by the runtime.
- **`Diagnostics/LoadProgress.swift`** is a thread-safe singleton progress reporter: the
  model-load path writes milestones and per-unit advances, and the UI polls
  `snapshot` to render a determinate bar.

The user-facing sampling knobs (temperature, repetition penalty, the fixed
top-k) are documented in the root
[Configuration Reference](../README.md#configuration-reference).

For the end-to-end lifecycle see
[PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md); for folder ownership read
[`Sources/DS4Core/README.md`](../Sources/DS4Core/README.md).
