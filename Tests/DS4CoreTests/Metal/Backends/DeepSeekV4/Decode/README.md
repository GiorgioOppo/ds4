# DeepSeek-V4 Decode Tests

Tests for `StreamingDecoder`, indexer selection, expert-cache allocation, layer
cache policy, and decode state transitions.

`ContextCapacityPolicyTests.swift` exercises the pure long-context policies
without requiring a Metal device: the live indexer boundary, attention-scratch
row calculation, geometric growth, and the configured hard cap.

Cover reset/reuse and cache hit/miss behavior in addition to token output. When
touching the raw-KV ring, compare runs longer than `nSWA` with the feature on
and off. Production-model smoke tests must skip clearly when their fixture is
not configured.
