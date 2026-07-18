# Backends

Implementazioni Metal specifiche per famiglia di modello. Runtime, tensori e
primitive GPU condivise restano nelle cartelle di primo livello di `DS4Metal`;
architettura, schema GGUF, pesi, stato KV, prefill e decode appartengono invece
al rispettivo backend.

## Backend

- [`Common/`](Common/README.md): regole del confine condiviso e future API di
  selezione a livello token/chunk.
- [`DeepSeekV4/`](DeepSeekV4/README.md): backend attualmente operativo.
- [`GLM52/`](GLM52/README.md): schema, riferimenti DSA e primitive Metal in
  costruzione; non contiene ancora un decoder eseguibile.
- [`Qwen/`](Qwen/README.md): spazio riservato; Qwen non è ancora supportato.

## Regole di modifica

La famiglia viene scelta al caricamento del modello, mai dentro il ciclo
per-layer o per-kernel. Non costruire contenitori universali pieni di campi
opzionali: ogni backend conserva pesi, scratch e snapshot con tipi concreti.
Promuovere una funzione nel livello comune solo quando semantica, layout e
vincoli di sincronizzazione coincidono realmente tra più backend.
