# GLM 5.2 compact DSA reference

This directory contains non-runnable policies and CPU correctness oracles for
GLM 5.2's compact DSA attention.

The compact cache stores, per live token, 512 KV-LoRA and 64 RoPE-tail values
for each of 78 normal layers. A 128-value indexer key is stored only for the 21
full-indexer layers: 0, 1, 2, then 6 through 74 every four layers. At F16 this
is 95,232 bytes per token.

`GLM52CompactDSACapacityPolicy` grows packed append-only slabs on demand rather
than allocating the complete logical context at session creation. A future GPU
implementation must preserve populated rows while adding slabs and must apply
the memory-budget check before allocation.

`GLM52IndexerCPUReference` implements the 32-by-128 weighted ReLU score, causal
deterministic top-k and IndexShare reuse. It is an oracle, not an optimized
decode path and not a declaration that the GLM runtime is ready. It is kept
separate from the sibling batch-Metal score contract: this F32 single-token
oracle owns finite-score validation, deterministic selection and reuse policy,
whereas the batch contract models F16 cache input and causal `-inf` masking.
