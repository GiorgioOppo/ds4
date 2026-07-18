# Formats

Binary formats and numeric conversions shared by model loading and
persistence.

## Structure

- [`GGUF/`](GGUF/README.md): mmap parser for the model file and its metadata.
- [`KVCheckpoint/`](KVCheckpoint/README.md): persistent KV cache format.
- [`Quantization/`](Quantization/README.md): f16/f32 and Q8_0 -> Q4_K on CPU.

## Flow and dependencies

`GGUFModel` validates and maps the model; `DS4Metal` uses the resulting
descriptors to create views over the weights. `KVCFile` serializes the
recurrent state outside the GGUF path. The CPU conversions are used when
preparing the quantized caches. The whole folder stays independent of Metal.

## Modification rules

Always validate bounds, overflow, alignment and endianness before reading. Do
not change persistent layouts without a version or migration. Avoid copies of
large payloads in the GGUF path.
