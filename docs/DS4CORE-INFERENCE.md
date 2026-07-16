# DS4Core inference-facing components

Pure inference-facing pieces that do not touch the GPU. Portable contracts are
kept separate from the frontend of each model backend.

- **`Tokenization/Backends/DeepSeekV4/DeepSeekV4Tokenizer.swift`** implements
  BPE with DeepSeek-V4 control tokens such as BOS/EOS, `<｜User｜>`, `<think>`,
  and `｜DSML｜`. Important entry points include `tokenizeRenderedChat` and
  `tokenText`; `Tokenizer` remains a compatibility alias.
- **`Conversation/Models/ConversationModels.swift`** defines `ToolSpec`,
  `ToolCall`, and `ChatTurn`; the files under
  `Conversation/Backends/DeepSeekV4/DSML` render chat/tool prompts and extract
  calls from generated DSML.
- **`Generation/Sampler.swift`** implements temperature, top-k/top-p, min-p, and repetition
  penalty sampling.
- **`Model/Common/ModelArchitecture.swift`** detects and describes the model
  family without constructing a decoder.
- **`Model/Backends/DeepSeekV4/DeepSeekV4Configuration.swift`** validates the
  DeepSeek profiles consumed by the current runtime.
- **`Diagnostics/LoadProgress.swift`** is a thread-safe singleton progress reporter: the
  model-load path writes milestones and per-unit advances, and the UI polls
  `snapshot` to render a determinate bar.

The user-facing sampling knobs (temperature, repetition penalty, the fixed
top-k) are documented in the root
[Configuration Reference](../README.md#configuration-reference).

For the end-to-end lifecycle see
[PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md); for folder ownership read
[`Sources/DS4Core/README.md`](../Sources/DS4Core/README.md).
