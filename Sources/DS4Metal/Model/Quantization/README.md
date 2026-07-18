# Model/Quantization

Description of the quantized formats accepted by the MoE kernels.

## Main files

- [`MoEQuant.swift`](MoEQuant.swift): enum of the supported quantizations and
  derived properties, including block size, bytes per row and dispatch
  parameters.

## Flow and dependencies

The loader translates the GGUF type into `MoEQuant`; gather, cache and kernels
use the same value to compute gate/up/down ranges and pick the pipeline.
The CPU conversions live in
[`DS4Core/Formats/Quantization`](../../../DS4Core/Formats/Quantization/README.md).

## Modification rules

A new case requires coordinated support in the GGUF parser, layout
computation, Swift wrapper and `.metal` kernel. Do not infer the type from the
bytes/elements ratio alone; fail on unvalidated combinations.
