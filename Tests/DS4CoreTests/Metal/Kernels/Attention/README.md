# Attention Kernel Tests

Tests for RoPE, KV compression, indexer scoring/pooling, sparse selection, flash
attention, and attention-output kernels.

Cover short and non-block-aligned sequence lengths, masks, top-k boundaries,
and numerical comparison with CPU attention. GPU-unavailable environments must
use `XCTSkip`, never silently return.

