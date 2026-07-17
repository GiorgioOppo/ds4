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

`GLM52AttentionCPUReference` is the oracle for the stage after selection: the
attention core over the compact cache rows. It keeps two evaluation orders —
`expanded` (upstream's textbook F32 reference: materialize per-head keys and
values from k_b/v_b, then attend) and `absorbed` (the kernel order: absorb the
query into k_b, score raw 512-wide cache rows, accumulate softmax in the
KV-LoRA domain, project through v_b once). Their tolerance-checked agreement is
the fixture the future `qk_lowrank`/`attention_indexed` Metal kernels must
match.
