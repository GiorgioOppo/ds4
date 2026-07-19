**English** | [Italiano](README.it.md)

# Tests/DS4CoreTests

Parity and unit tests for `DS4Core`, `DS4Metal`, and `DS4Engine` with XCTest.
Each file targets a kernel or component and compares the output against a
reference, often a CPU-faithful implementation.

Covered areas include Metal kernels (`Graph*`, `Metal*`, `MoE`),
`StreamingDecoder`, the GGUF loader, `Half`, `KVCFile`, tokenizers, samplers,
DeepSeek and GLM chat/tool protocols, the GLM schema and DSA layout,
`SSDCachePlan`, the disk KV store, the distributed protocol, MCP, the project
cache, the tool registry, and downloader utilities such as `sha256Hex` and the
target map.

When touching a numerical invariant, such as the **raw-KV ring**, add a test that
generates more than `nSWA` tokens with and without the feature and asserts
equality.

## Directory map

- [`Core/`](Core/README.md): CPU-only deterministic unit tests.
- [`Metal/`](Metal/README.md): kernels, graphs, decoder, loader, and runtime.
- [`Engine/`](Engine/README.md): services, persistence, protocols, projects, and
  tools.

Tests follow production ownership: add a new case beside the component's area,
not at this directory root. Use temporary files and injected dependencies;
never rely on the user's credentials, repositories, downloaded model, or
network access. GPU skip conventions are documented in
[`../METAL-TESTS.md`](../METAL-TESTS.md).
