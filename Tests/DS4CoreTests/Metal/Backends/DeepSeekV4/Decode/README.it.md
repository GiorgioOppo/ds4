# Test di decode DeepSeek-V4

Test per `StreamingDecoder`, selezione dell'indexer, allocazione della cache
degli esperti, policy della cache dei layer e transizioni di stato del decode.

`ContextCapacityPolicyTests.swift` esercita le policy pure per i contesti
lunghi senza richiedere un device Metal: il confine dell'indexer live, il
calcolo delle righe dello scratch di attenzione, la crescita geometrica e
l'hard cap configurato.

Copri il comportamento di reset/riuso e di cache hit/miss oltre all'output dei
token. Quando tocchi il ring raw-KV, confronta run più lunghe di `nSWA` con la
feature attiva e disattivata. Gli smoke test sui modelli di produzione devono
saltare in modo chiaro quando la loro fixture non è configurata.
