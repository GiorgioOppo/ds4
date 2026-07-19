**English** | [Italiano](README.it.md)

# Kernels

Swift wrappers for the Metal compute pipelines. Each file binds a family of
`.metal` functions, validates the essential arguments and encodes the
dispatch.

## Structure

- [`Attention/`](Attention/README.md): flash attention, RoPE and sparse indexer.
- [`Compression/`](Compression/README.md): KV compressor and HyperConnections.
- [`Dense/`](Dense/README.md): dense and quantized matvec/matmul.
- [`MoE/`](MoE/README.md): router and expert FFN.
- [`Tensor/`](Tensor/README.md): generic tensor transformations.

The authoritative sources are in `metal/*.metal`; the embedded copy is
described in [`Runtime/Generated`](../Runtime/Generated/README.md).

## Flow and rules

The [`Graph`](../Graph/README.md) calls these wrappers with `GPUTensor` and
command buffers. A wrapper does not decide the model sequence and must not
perform hidden CPU waits. After a kernel change: modify the `.metal` source,
run `make embed-kernels`, update the Swift signature/dispatch and add tests
for shape, dtype, non-zero offset and threadgroup limits.
