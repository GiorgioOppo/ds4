# Inference/API

Contains the public `Sendable` types used by clients of the engine.

## Main types

- `ChatRole`: logical role of a turn.
- `DS4ThinkMode`: reasoning mode exposed by the application.
- `SamplingParams`: temperature, top-k/top-p/min-p, seed and repetition
  penalty.
- `ModelInfo`: concise description of the loaded model; keeps
  `routedQuantBits` for compatibility and adds architecture, display name,
  quantization summary and runtime capabilities.
- `GenEvent`: stream of reasoning, text, tool calls and progress.
- `InferenceError`: application errors presentable to the caller.

## Dependencies and flow

The types depend on Foundation and, where needed, on the portable models of
`DS4Core`. They are produced by [`Service`](../Service/README.md) and consumed
by GUI, server and benchmarks without exposing Metal objects.

## Extension

Keep the types independent of SwiftUI and `Metal`. A new event must
have clear semantics in the stream and every consumer must handle it
explicitly; avoid state strings where an enum case is needed.
