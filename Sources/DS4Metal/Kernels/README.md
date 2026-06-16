# DS4Metal/Kernels

Un wrapper Swift per ogni kernel Metal: prepara gli argomenti, sceglie la pipeline e fa il dispatch. I **sorgenti** dei kernel sono in `metal/*.metal` (source of truth), embeddati nel binario via `make embed-kernels` → `Runtime/KernelSources.swift`.

Gruppi principali:
- **MoE** — `MetalMoE`, `MetalMoEFused` (matvec esperti fuso SwiGLU-pair + down-sum6), `MetalRouter`, `MetalSparseSelect`, `MetalArgsort`.
- **Attenzione** — `MetalFlashAttn`, `MetalAttnOutLow`, `MetalRoPE`, `MetalKVCompress`, `MetalCompressor`, `MetalIndexerScore`, `MetalIndexerPool`.
- **Algebra/utility** — `MetalDense`, `MetalMatmulMM`, `MetalNorm`, `MetalSoftmax`, `MetalSumRows`, `MetalGLU`, `MetalUnary`, `MetalGetRows`/`SetRows`, `MetalCopy`, `MetalConcat`, `MetalRepeat`, `MetalBin`, `MetalHCSplit`, `MetalHyperConnections`.

Aggiungere/modificare un kernel: edita `metal/*.metal`, poi `make embed-kernels`, e aggiorna/aggiungi il wrapper qui.
