**English** | [Italiano](README.it.md)

# GLM 5.2 layer execution tests

GPU-vs-oracle suites for the layer composition: dense and sparse first-token
layers against `GLM52LayerCPUReference` on the dequantized weights, routing
equality (selection exact, weights within tolerance), the two-layer forward
chain with the output head, and input-shape rejections. Fixtures follow the
MoE suite discipline — Q8_0 from the shared test quantizer, synthetic Q4_K
expert rows, the quantized bytes as the single source of truth for both
sides. Skips without a Metal device.
