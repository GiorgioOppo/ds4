# metal

Sorgenti dei kernel Metal (`.metal`) — la **source of truth**. Vengono embeddati nel binario via `make embed-kernels` (`scripts/embed_kernels.sh`) → `Sources/DS4Metal/Runtime/KernelSources.swift`, così il runtime non ha bisogno di una cartella kernel su disco (funziona in SwiftPM, nell'`.xcodeproj` e in una `.app` spedita).

File principali (per dimensione/peso): `moe.metal` (matvec MoE per tutti i quant), `flash_attn.metal`, `dense.metal`, `dsv4_misc.metal`, `dsv4_hc.metal`, `dsv4_kv.metal`, `dsv4_rope.metal`, + utility (norm, softmax, argsort, unary…).

**Workflow**: edita un `.metal` → `make embed-kernels` → aggiorna/aggiungi il wrapper in `Sources/DS4Metal/Kernels/`. Tieni l'ordine in sync con `MetalRuntime.kernelFiles`.
