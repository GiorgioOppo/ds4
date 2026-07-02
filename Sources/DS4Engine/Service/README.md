# DS4Engine/Service

Il servizio di inferenza e la sua persistenza.

- **`InferenceService.swift`** — attore centrale. Possiede il `StreamingDecoder`; rende `send`/`provideToolResults`/`complete` come stream di eventi; gestisce il riuso KV append-only (`committedIds`), il benchmark, lo switch agente + usage imatrix, e i **sub-agent** (`runSubAgent`: snapshot/restore del KV main attorno a un contesto isolato).
- **`DiskKVStore.swift`** — KV cache su disco (modello `ds4_kvstore`): checkpoint per-prefisso, restore a freddo, eviction sotto budget. Usato anche per le KV cache content-key dei sub-agent.
- **`Diagnostics.swift`** — dump token / chat template (tokenizer nativo, niente sottoprocessi).

`InferenceService` è grosso: il sub-agent sarebbe un candidato a `SubAgent.swift` (extension) una volta sciolti i `private`.
