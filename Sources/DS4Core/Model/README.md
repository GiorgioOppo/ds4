**English** | [Italiano](README.it.md)

# Model

Portable architecture detection and per-backend isolated configurations.

## Structure

- [`Common/`](Common/README.md): canonical identifier, family, capabilities,
  descriptor, and detection from `general.architecture`.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): shape and
  validation of DeepSeek V4 metadata, with compatible aliases for the
  historical APIs.
- [`Backends/GLM52/`](Backends/GLM52/README.md): GLM 5.2 geometry and strict
  validation of the `glm-dsa` namespace.
- [`Backends/Qwen/`](Backends/Qwen/README.md): documented extension point;
  the Qwen backend is not yet implemented.

## Flow and dependencies

The detector classifies the architecture first; only then may the matching
backend read and validate its own metadata namespace. The DeepSeek-V4
kernel-specific constants remain in
[`DS4Metal/Backends/DeepSeekV4/Architecture`](../../DS4Metal/Backends/DeepSeekV4/Architecture/README.md).

## Modification rules

Do not interpret a recognized but unimplemented architecture through the
DeepSeek backend. New variants must have explicit validation and checked
derived values; do not duplicate here constants already read from the GGUF.
