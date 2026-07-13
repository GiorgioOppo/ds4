# Inference/Diagnostics

Fornisce diagnostica read-only sul GGUF senza avviare il decoder Metal.

## Componente

`Diagnostics.swift` apre metadati e tokenizer per:

- mostrare ID e testo dei token;
- ispezionare `tokenizer.chat_template`;
- verificare token speciali e markup DSML;
- confrontare dichiarazioni tool complete e compatte.

## Dipendenze e flusso

Dipende da `DS4Core` e Foundation. Gli output sono stringhe destinate a GUI,
log o test; nessuno stato del servizio viene modificato.

## Estensione

Preferire controlli deterministici e senza side effect. Una diagnosi che misura
la GPU appartiene a [`Benchmark`](../Benchmark/README.md), non qui.
