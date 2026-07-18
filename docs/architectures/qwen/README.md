# Qwen backend — groundwork

Qwen is currently a **recognized but not runnable** family. This folder
documents the boundary prepared for adding the backend without introducing
Qwen conditionals into the DeepSeek decoder.

## Already in place

- family identification from `general.architecture`;
- dedicated error when the backend is not yet implemented;
- separate folders in the Core, Metal and Engine layers;
- model description and capabilities consumable by demo and GUI;
- separation of the DeepSeek components ahead of adding new kernels.

## Not yet implemented

- choice of the reference Qwen GGUF variant;
- tokenizer, special tokens and chat template;
- tensor mapping and shape validation;
- prefill, Metal decoder and KV cache;
- reasoning and tool-call format;
- UI presets and numerical diagnostics;
- checkpoints, distributed inference and certified benchmarks.

Until these elements are complete, loading a Qwen GGUF must stop before
decoder construction with a message stating that the model was recognized but
the backend is not yet available.

## First implementation step

Before writing kernels, pick a single small, representative GGUF file, record
the exact value of `general.architecture`, the tokenizer metadata and the
tensor list, then pin CPU golden tests. Only from that contract are shape,
loader and Metal graphs derived.
