# Decode

Orchestrazione dell'inferenza ricorrente: inizializzazione, prefill, forward di
un token, gestione KV, output head e diagnostica.

## Struttura

- [`Execution/`](Execution/README.md): stato del decoder e percorso per-layer.
- [`Prefill/`](Prefill/README.md): ingestione layer-major di più token.
- [`Generation/`](Generation/README.md): embedding/output head e generazione.
- [`State/`](State/README.md): buffer scratch riutilizzabili.
- [`KV/`](KV/README.md): snapshot e ripristino dello stato ricorrente.
- [`Attention/`](Attention/README.md): selezione top-k CPU dell'indexer.
- [`Cache/`](Cache/README.md): cache LRU e statistiche degli expert.
- [`Diagnostics/`](Diagnostics/README.md): profilo temporale e I/O.
- [`Reference/`](Reference/README.md): implementazione di riferimento per parità.

## Flusso

Il factory prepara runtime, pesi e cache. Il prefill attraversa il prompt in
chunk layer-major, aggiornando lo stesso stato usato dal decode. La generazione
esegue poi un forward per token: embedding -> layer -> attention/KV -> router e
FFN -> output head. Gli snapshot permettono di sospendere e riprendere il flusso.

Il raw KV è una finestra circolare/lineare limitata a `nSWA`; il contesto più
vecchio sopravvive nelle righe compresse NSA. Le opzioni runtime sono nella
[Configuration Reference](../../../README.md#configuration-reference).

## Regole di modifica

Prefill e decode devono conservare la stessa semantica ricorrente. Ogni percorso
asincrono deve definire ownership, punto di attesa e cancellazione. Una modifica
al layout KV richiede aggiornamento coordinato di snapshot, checkpoint e test.
