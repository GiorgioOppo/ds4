**English** | [Italiano](README.it.md)

# MoE Kernel Tests

Coverage for routing and fused/non-fused expert execution across Q2_K, Q4_K,
IQ2_XXS, and other supported weight layouts.

Tests should compare route selection, normalized weights, and final activations
with CPU references. Include active-expert boundaries and non-default
simdgroup counts where results are required to remain identical.

