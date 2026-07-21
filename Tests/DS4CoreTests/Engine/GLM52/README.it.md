[English](README.md) | **Italiano**

# Test Engine — servizio GLM 5.2

Test per le superfici GLM 5.2 a livello DS4Engine (`GLM52ChatService` e i suoi
helper di benchmark/accuratezza), distinti dai test del motore Metal sotto
`Metal/Backends/GLM52`.

- `GLM52AccuracyCandidatesTests.swift` fissa lo scorer Top-K a selezione
  parziale usato dal benchmark Correctness GLM: ordine decrescente, pareggi
  vinti dall'id più basso (la regola dell'argmax greedy) ed equivalenza con
  un ordinamento completo su un vocabolario grande.
- `GLM52DiskKVStoreTests.swift` copre lo store di checkpoint per prefisso su
  file GKV1 sintetici: lookup del prefisso più lungo, dedup/minTokens,
  policy dell'intervallo di store, eviction supersede sotto budget in token,
  adozione del `state.glmkv` legacy, ricostruzione dell'indice e tolleranza
  ai file estranei.

Regole: i test qui non devono richiedere un GGUF reale né un device Metal;
ciò che esegue il motore sta con i test GLM lato Metal.
