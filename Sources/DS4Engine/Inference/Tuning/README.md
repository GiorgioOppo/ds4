# Inference/Tuning

Collects and persists the expert-routing profile used to warm the
slot cache and observe hits/misses.

## Component

`InferenceService+Tuning.swift` manages usage files per model/agent pair,
selects an initial profile, exposes `TuningInfo` and `ModelInfo`, and
estimates KV memory based on the active configuration.

## Flow and dependencies

The profile is loaded during initialization of the
[`Service`](../Service/README.md), updated by generation and saved at the end
of a turn. The counters come from `DS4Metal`; the files are application data
in Application Support.

## Extension

Version incompatible persistent formats, always keep profiles of different
models or agents separate, and never use these statistics to silently change
the model's numerical quality.
