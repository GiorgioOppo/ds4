# DeepSeekV4/Decode/Attention

Logica CPU di supporto alla selezione sparsa dell'attenzione NSA.

## File principali

- [`IndexerSelect.swift`](IndexerSelect.swift): top-k a heap con ordinamento
  deterministico per punteggio decrescente e indice crescente a parità.

## Flusso e dipendenze

I kernel producono gli score dell'indexer; quando è usato il fallback CPU, gli
score vengono letti e `IndexerSelect` restituisce gli indici delle righe KV da
passare all'attenzione sparsa. Il percorso GPU equivalente si trova nei wrapper
[`Kernels/Attention`](../../../../Kernels/Attention/README.md).

## Regole di modifica

Conservare lo stesso insieme e tie-break del sort completo, inclusi NaN e
contesti più corti di k. Misurare separatamente complessità CPU, sincronizzazione
GPU e costo di readback.
