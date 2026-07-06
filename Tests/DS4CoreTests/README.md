# Tests/DS4CoreTests

Parity and unit tests for `DS4Core`, `DS4Metal`, and `DS4Engine` with XCTest.
Each file targets a kernel or component and compares the output against a
reference, often a CPU-faithful implementation.

Covered areas include Metal kernels (`Graph*`, `Metal*`, `MoE`),
`StreamingDecoder`, the GGUF loader, `Half`, `KVCFile`, the tokenizer, sampler,
and DSML chat/tool parsing, `SSDCachePlan`, the disk KV store, the distributed
protocol, MCP, the project cache, the tool registry, and downloader utilities
such as `sha256Hex` and the target map.

When touching a numerical invariant, such as the **raw-KV ring**, add a test that
generates more than `nSWA` tokens with and without the feature and asserts
equality.
