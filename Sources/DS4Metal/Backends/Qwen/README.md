**English** | [Italiano](README.it.md)

# Qwen backend

Placeholder for future Qwen support. The family is recognized in the project
structure, but no Qwen loader, decoder, or kernels exist yet, and no Qwen
model must be accepted by the current runtime.

Before implementing, the GGUF variant must be chosen explicitly, its metadata
and tensor naming validated, and concrete types defined for weights, scratch,
KV cache, attention, and output head. The backend must not reuse the MLA
attention, the HyperConnections, the NSA compressors, or the DeepSeek-V4
router as a fallback.
