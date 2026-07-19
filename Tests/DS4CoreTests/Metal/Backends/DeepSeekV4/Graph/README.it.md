[English](README.md) | **Italiano**

# DeepSeek-V4 Graph Tests

Composizioni `GraphContext` specifiche di DeepSeek-V4: Q/KV latente, attention
MLA, output low-rank, compressore NSA, router top-6, routed MoE,
HyperConnections e decoder/layer completi.

I test delle primitive riutilizzabili rimangono in
[`Metal/Graph`](../../../Graph/README.it.md); gli edge case dei singoli kernel
rimangono in `Metal/Kernels`.
