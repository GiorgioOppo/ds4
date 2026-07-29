**English** | [Italiano](README.it.md)

# Laguna engine

`LagunaResidentModel` is the first-cut resident engine: every validated
tensor of the official Q8_0-signal / Q4_K-routed recipe is uploaded once into
shared `MTLBuffer`s (Laguna requires full residency upstream — no SSD
streaming, sidecars or expert cache), each layer owns an F16 ring KV cache
(512 rows on sliding-window blocks, the configured context on full-attention
blocks), and the per-token graph mirrors `laguna_graph_forward_token`:
RMSNorm → paired Q8_0 matvecs (Q/K, V/gate) → per-head norm/RoPE → ring
store → gated GQA attention → output projection and residual → dense or
routed FFN. It dispatches the shared GLM primitives where upstream shares
them (`kernel_glm52_rms_norm_f32`, `kernel_glm52_matvec_pair_sg`,
`kernel_glm52_router_select` with 10 active experts, the `glm52_moe` K-quant
matvecs) plus the Laguna kernels beside them. The router selection is read
back on the host to address expert slabs, like the GLM chained decode.

Routed experts may be Q2_K, Q3_K or Q4_K per layer (coherent, as the schema
guarantees), so both the official Q4_K_M file and the mixed
RoutedQ2_K-Last27Q3_K file run; the Q3_K dot helpers live beside the other
K-quants in `metal/glm5.2/glm52_quant.metal`. Deliberate scope limits of
this cut, refused with distinct errors at load: the legacy F16/Q4_K recipe
(its matvec paths are not wired) and batched prefill (prompts run
token-by-token through the decode path — correct, not fast).
`LagunaResidentModelOptions.layerCount` truncates the stack from the front
for bring-up runs.

`LagunaResidentModelOptions.expertStreaming` is an opt-in, declared
divergence from upstream (which mandates full residency for Laguna): the
Q8_0 signal path stays resident (~5 GiB) and the slabs of the 10 selected
routed experts are copied from the mmap per token after the host router
readback — same kernels, same bytes, ~1.6 GB of reads per token. It exists
so 32 GB machines can run the 45 GiB file at all; expect low single-digit
tok/s. `DecodeProfile` (`profileReport()`) reports the per-phase cost,
counting the slab reads as gather IO.

`LagunaRuntimeGate.enabled` stays `false` until this engine passes
end-to-end logits parity against the reference C engine on real weights;
selection, catalog availability and the demo dispatch all key off that one
constant.
