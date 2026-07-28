[English](README.md) | **Italiano**

# Test dello schema tensori Laguna

Copertura basata su record di `LagunaTensorSchema` senza mappare una fixture
da 60+ GB: entrambe le ricette pubblicate (signal path Q8_0 e legacy
Q4_K/F16), il file misto con esperti Q2_K/Q3_K, la coerenza per-layer del tipo
instradato, la regola del marcatore di layout, le larghezze query per-layer
48/72, le viste solo-layer identificate dall'attention Q e la tassonomia degli
errori missing/partial/duplicate.
