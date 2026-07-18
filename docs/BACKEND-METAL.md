# Metal Backend

This guide describes the boundaries of the shared GPU runtime and the correct
path for modifying a backend, a wrapper, or a kernel. The DeepSeek backend
formulas are covered in depth in
[ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.md); the multi-model separation
is defined in
[ARCHITETTURE-SUPPORTATE.md](ARCHITETTURE-SUPPORTATE.md).

## Backend layers

```text
Concrete backend (today DeepSeekV4/StreamingDecoder;
GLM52 remains in isolated, unconnected tranches)
             |
             v
decoder, KV state and the backend's weight provider
             |
             v
Graph/Core + Graph/Operations
             |
             v
Swift wrappers in Kernels/<Area>
             |
             v
Metal functions in metal/*.metal
```

| Area | Responsibility |
|---|---|
| `Runtime/Core` | device, command queue, pipeline compilation and `GPUTensor` |
| `Runtime/Generated` | generated copy of the Metal sources embedded in the binary |
| `Kernels` | Swift bindings, arguments, dispatch sizes and buffers |
| `Graph` | operation composition and command buffer management |
| `Model/Quantization` | genuinely shared quantization descriptors |
| `Backends/DeepSeekV4` | shape, tensors, streaming, experts, MTP, decoder, prefill and DeepSeek KV |
| `Backends/Qwen` | documented placeholder; no current GPU implementation |
| `metal` | authoritative GPU implementation |

Selection happens before the decoder is built. The per-layer loop uses the
backend's concrete type and does not consult a registry or a dynamic protocol.

## Runtime and tensors

`MetalRuntime` selects the device, creates the queue, compiles the embedded
source and keeps pipelines by name. `GPUTensor` associates an `MTLBuffer` with
shape, type, stride and offset. A tensor is a memory view: anyone introducing
subviews must verify that the wrapper in use correctly propagates
`byteOffset`.

Metal resources are managed by ARC, but temporal correctness remains
explicit: a staging buffer cannot be reused before the command buffer that
reads it has completed.

## Command buffers and synchronization

`GraphContext` accumulates related dispatches and decides when to `commit` and
when to wait for the result. Every CPU/GPU wait on the per-token path is
potentially expensive. The existing fusions remove intermediate passes without
moving application policy into the kernels.

Before removing a synchronization, verify:

- which buffer the previous dispatch produces;
- who reads it and on which command buffer;
- whether there is a CPU read of the results;
- when a staging or cache slot may be overwritten;
- whether the numeric accumulation order must stay identical.

## Kernel wrappers

Wrappers are grouped by function:

- `Kernels/Attention` — RoPE, KV compression, flash attention and indexer;
- `Kernels/Compression` — compressors and hyper-connections;
- `Kernels/Dense` — dense projections and matmul;
- `Kernels/MoE` — router and quantized experts;
- `Kernels/Tensor` — general primitives, norms, softmax and copies.

A wrapper must validate buffer shape and capacity, create an argument struct
with a Metal-compatible layout, set every offset and choose a grid consistent
with the kernel's limits. It must not decide how many experts to activate,
which prompt to process, or which cache strategy to use.

## Embedded sources

The editable files are the vendored kernels grouped by architecture:
`metal/deepseek/*.metal` (DeepSeek V4 plus the shared generic ops) and
`metal/glm5.2/*.metal` (GLM 5.2). The file
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift` is generated and must
not be edited manually.

Mandatory workflow:

```sh
make embed-kernels
swift test --disable-sandbox
swift build -c release --product DS4Demo --disable-sandbox
```

`make embed-kernels` concatenates the sources in the same order the runtime
expects. If a function signature changes, the Swift wrapper and tests must be
updated in the same change.

## Weights and quantization

The backend uses several representations depending on the tensor:

- F32/F16 for state, caches and some projections;
- Q8_0 for many dense weights;
- Q4_K, Q2_K and IQ2_XXS for the experts;
- derived Q4/Q8 caches enabled only by their toggles.

In the DeepSeek backend, `LayerWeights` carries the effective quantization of
each individual layer. Do not
infer the format from a global setting when the GGUF may be mixed
precision. `GGUFWeights` validates type, shape, offset and size before
exposing a tensor to the graph.

## Streaming and unified memory

Apple Silicon shares CPU/GPU memory, but that does not eliminate the costs of
paging, compression and I/O. The main strategies are:

- mmap for zero-copy views;
- dense staging with `pread` reads outside the page cache;
- best-effort `mlock` for hot buffers;
- expert slot-cache;
- contiguous bundle to reduce seeks and the number of reads;
- MetalIO with automatic fallback when the real bandwidth is insufficient.

MetalIO is not arbitrary SSD access from a shader: it is resource loading
coordinated by the Metal runtime. The command processor keeps using valid
Metal buffers, and the path retains a CPU/`pread` fallback.

## Numeric rules

Optimizations fall into three categories, which must be declared in the
tests and in the documentation:

1. **bit-identical**, same order and same results;
2. **mathematically equivalent**, but with a possible difference of a few ulps;
3. **lossy**, for requantization or reducing the model's work.

A throughput increase is not enough to accept a kernel. At minimum you need:

- comparison with the CPU implementation or the previous path;
- coverage of edge sizes and supported quantizations;
- NaN/Inf and bounds checks;
- comparison of the generated text with fixed seed and prompt;
- measurement after warm-up without invasive profiling.

The dormant risks already identified are listed in
[metal/README.md](../metal/README.md); do not enable a path marked as
unreached without first resolving or validating its preconditions.

## Adding a new operation

1. First establish whether the operation is shared or belongs to a single
   backend; then choose the domain (`Attention`, `Compression`, `Dense`, `MoE`,
   `Tensor`).
2. Add or modify the kernel in the appropriate `.metal` file.
3. Create a focused wrapper under `Sources/DS4Metal/Kernels/<Area>`.
4. Compose it in the owning backend; use `Graph/Operations` only for a
   primitive with a genuinely common contract, free of GUI or service
   state.
5. Add a kernel test with a CPU reference.
6. Add a graph test if the operation participates in a chain.
7. Regenerate `KernelSources.swift` and build the demo and the app.

## Diagnostics

`DecodeProfile` separates expert I/O, route/attention, FFN and overhead. Fine
profiling can add waits and alter throughput; use it to identify proportions
and bottlenecks, then measure the final speed with profiling disabled.

See also:

- [VALUTAZIONE-DEMO-PERF.md](VALUTAZIONE-DEMO-PERF.md)
- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md)
- [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md)
