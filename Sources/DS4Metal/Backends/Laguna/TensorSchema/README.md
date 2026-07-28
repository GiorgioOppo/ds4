**English** | [Italiano](README.it.md)

# Laguna tensor schema

`LagunaTensorSchema` validates the tensor directory of a Laguna S 2.1 GGUF
without reading payloads, ported from `weights_validate_laguna_layout` in the
reference `laguna-s2.1` branch.

Two coherent recipes are accepted, identified from the embedding tensor (or
the first layer's attention Q for layer-only views): the current official
Q4_K_M file with **Q8_0 signal-path** weights, and the earlier **legacy**
recipe with F16 attention and Q4_K/Q6_K signal weights. Routed experts may be
Q4_K, Q3_K or Q2_K and may differ between layers (the published mixed file
uses Q2_K on early MoE layers and Q3_K on the last 27), but the three routed
projections of one layer must be coherent; the legacy recipe additionally
accepts Q6_K down projections next to Q4_K gate/up. Everything else —
per-layer 48/72 query-head widths, GQA K/V widths, dense and shared FFN
shapes, router tensors, output head — must match the exact S 2.1 geometry.
