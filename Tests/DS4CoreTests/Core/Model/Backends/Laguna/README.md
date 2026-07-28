**English** | [Italiano](README.it.md)

# Laguna configuration tests

These metadata-only GGUF fixtures verify the exact Laguna S 2.1 shape without
requiring the full model.  Tests cover the per-layer 48/72 head-count
alternation and its sliding-window rule, the YaRN requirement, architecture
isolation, shape mismatches and expert normalization.
