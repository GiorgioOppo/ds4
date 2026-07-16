# Model

Tipi di modello condivisi tra i backend Metal. Le descrizioni architetturali,
gli schemi tensor GGUF, lo streaming e le cache specifiche vivono sotto
[`Backends/`](../Backends/README.md).

## Struttura

- [`Quantization/`](Quantization/README.md): metadati dei layout MoE quantizzati.

## Flusso

Il codice in questa cartella non deve assumere nomi tensor, forma della KV cache,
numero di head o strategia di routing di una singola famiglia. Il backend
DeepSeek-V4 mantiene il proprio modello in
[`Backends/DeepSeekV4`](../Backends/DeepSeekV4/README.md).

## Regole di modifica

Promuovere qui un tipo soltanto quando almeno due backend ne condividono davvero
semantica e layout. Fallire esplicitamente su forme o quantizzazioni non
supportate; non usare fallback che producano logits plausibili ma errati.
